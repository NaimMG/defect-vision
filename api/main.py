import io
import base64
import numpy as np
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.patchcore import PatchCoreInference
from src.yolov8_inference import YOLOv8Inference

# ── Chemins modèles ──────────────────────────────────────────────
MODELS_DIR       = Path(__file__).parent.parent / "models"
YOLO_MODEL_PATH  = MODELS_DIR / "yolov8_bottle" / "best.pt"
PC_MODEL_PATH    = MODELS_DIR / "patchcore_bottle" / "memory_bank.pt"

# ── App FastAPI ──────────────────────────────────────────────────
app = FastAPI(
    title       = "DefectVision API",
    description = "Industrial defect detection with YOLOv8 & PatchCore",
    version     = "1.0.0",
)

# ── Chargement des modèles ───────────────────────────────────────
yolo_model      = None
patchcore_model = None


@app.on_event("startup")
async def load_models():
    global yolo_model, patchcore_model
    if YOLO_MODEL_PATH.exists():
        yolo_model = YOLOv8Inference(str(YOLO_MODEL_PATH))
        print(f"✅ YOLOv8 chargé : {YOLO_MODEL_PATH}")
    else:
        print(f"⚠️  YOLOv8 non trouvé : {YOLO_MODEL_PATH}")

    if PC_MODEL_PATH.exists():
        patchcore_model = PatchCoreInference(str(PC_MODEL_PATH))
        print(f"✅ PatchCore chargé : {PC_MODEL_PATH}")
    else:
        print(f"⚠️  PatchCore non trouvé : {PC_MODEL_PATH}")


# ── Utils ────────────────────────────────────────────────────────
def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def heatmap_to_base64(heatmap: list) -> str:
    arr       = np.array(heatmap)
    arr_norm  = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    colored   = (cm.jet(arr_norm)[:, :, :3] * 255).astype(np.uint8)
    img       = Image.fromarray(colored).resize((224, 224), Image.NEAREST)
    return image_to_base64(img)


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name"   : "DefectVision API",
        "version": "1.0.0",
        "models" : {
            "yolov8"    : yolo_model is not None,
            "patchcore" : patchcore_model is not None,
        },
        "endpoints": [
            "POST /predict/yolov8",
            "POST /predict/patchcore",
            "POST /predict/combined",
            "GET  /health",
        ]
    }


@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "yolov8"    : yolo_model is not None,
        "patchcore" : patchcore_model is not None,
    }


@app.post("/predict/yolov8")
async def predict_yolov8(file: UploadFile = File(...)):
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 model not loaded")

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    result   = yolo_model.predict(image)

    return JSONResponse({
        "model"        : "yolov8",
        "n_detections" : result["n_detections"],
        "detections"   : result["detections"],
        "annotated_image_b64": image_to_base64(result["annotated_image"]),
    })


@app.post("/predict/patchcore")
async def predict_patchcore(file: UploadFile = File(...)):
    if patchcore_model is None:
        raise HTTPException(status_code=503, detail="PatchCore model not loaded")

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    result   = patchcore_model.predict(image)

    return JSONResponse({
        "model"        : "patchcore",
        "score"        : result["score"],
        "threshold"    : result["threshold"],
        "is_defect"    : result["is_defect"],
        "heatmap_b64"  : heatmap_to_base64(result["heatmap"]),
    })


@app.post("/predict/combined")
async def predict_combined(file: UploadFile = File(...)):
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    response = {}

    if yolo_model:
        yolo_result = yolo_model.predict(image)
        response["yolov8"] = {
            "n_detections"       : yolo_result["n_detections"],
            "detections"         : yolo_result["detections"],
            "annotated_image_b64": image_to_base64(yolo_result["annotated_image"]),
        }

    if patchcore_model:
        pc_result = patchcore_model.predict(image)
        response["patchcore"] = {
            "score"      : pc_result["score"],
            "threshold"  : pc_result["threshold"],
            "is_defect"  : pc_result["is_defect"],
            "heatmap_b64": heatmap_to_base64(pc_result["heatmap"]),
        }

    return JSONResponse(response)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)