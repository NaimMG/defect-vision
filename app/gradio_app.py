import io
import base64
import requests
import gradio as gr
import numpy as np
import matplotlib.cm as cm
from PIL import Image

API_URL = "http://localhost:8000"


def base64_to_image(b64_string: str) -> Image.Image:
    data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(data))


def predict_yolov8(image: Image.Image):
    if image is None:
        return None, "❌ Aucune image fournie"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict/yolov8",
            files={"file": ("image.png", buffer, "image/png")},
            timeout=30,
        )
        result = response.json()
    except Exception as e:
        return None, f"❌ Erreur API : {e}"

    annotated = base64_to_image(result["annotated_image_b64"])
    n         = result["n_detections"]
    dets      = result["detections"]

    if n == 0:
        info = "✅ Aucun défaut détecté"
    else:
        info = f"⚠️ {n} défaut(s) détecté(s)\n"
        for i, d in enumerate(dets):
            info += f"\n  #{i+1} — {d['label']} (conf: {d['confidence']:.2f})"
            info += f"\n       bbox: ({d['x1']:.0f}, {d['y1']:.0f}) → ({d['x2']:.0f}, {d['y2']:.0f})"

    return annotated, info


def predict_patchcore(image: Image.Image):
    if image is None:
        return None, None, "❌ Aucune image fournie"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict/patchcore",
            files={"file": ("image.png", buffer, "image/png")},
            timeout=60,
        )
        result = response.json()
    except Exception as e:
        return None, None, f"❌ Erreur API : {e}"

    heatmap   = base64_to_image(result["heatmap_b64"])
    score     = result["score"]
    threshold = result["threshold"]
    is_defect = result["is_defect"]

    # Overlay heatmap sur image originale
    img_resized  = image.resize((224, 224)).convert("RGB")
    heat_resized = heatmap.resize((224, 224)).convert("RGB")
    img_arr      = np.array(img_resized) / 255.0
    heat_arr     = np.array(heat_resized) / 255.0
    overlay_arr  = np.clip(0.55 * img_arr + 0.45 * heat_arr, 0, 1)
    overlay      = Image.fromarray((overlay_arr * 255).astype(np.uint8))

    status = "🔴 DÉFAUT DÉTECTÉ" if is_defect else "🟢 NORMAL"
    info   = f"{status}\n\nScore     : {score:.4f}\nSeuil     : {threshold:.4f}\nDécision  : {'Défaut' if is_defect else 'Normal'}"

    return heatmap, overlay, info


def predict_combined(image: Image.Image):
    if image is None:
        return None, None, None, "❌ Aucune image fournie"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict/combined",
            files={"file": ("image.png", buffer, "image/png")},
            timeout=60,
        )
        result = response.json()
    except Exception as e:
        return None, None, None, f"❌ Erreur API : {e}"

    # YOLOv8
    yolo_img = None
    yolo_info = "YOLOv8 non disponible"
    if "yolov8" in result:
        yolo_img  = base64_to_image(result["yolov8"]["annotated_image_b64"])
        n         = result["yolov8"]["n_detections"]
        yolo_info = f"{'⚠️ ' + str(n) + ' défaut(s)' if n > 0 else '✅ Aucun défaut'}"

    # PatchCore
    pc_overlay = None
    pc_info    = "PatchCore non disponible"
    if "patchcore" in result:
        heatmap   = base64_to_image(result["patchcore"]["heatmap_b64"])
        score     = result["patchcore"]["score"]
        threshold = result["patchcore"]["threshold"]
        is_defect = result["patchcore"]["is_defect"]

        img_r    = image.resize((224, 224)).convert("RGB")
        heat_r   = heatmap.resize((224, 224)).convert("RGB")
        ov_arr   = np.clip(0.55 * np.array(img_r)/255 + 0.45 * np.array(heat_r)/255, 0, 1)
        pc_overlay = Image.fromarray((ov_arr * 255).astype(np.uint8))
        pc_info  = f"{'🔴 DÉFAUT' if is_defect else '🟢 NORMAL'} — score={score:.3f}"

    summary = f"YOLOv8   : {yolo_info}\nPatchCore: {pc_info}"

    return yolo_img, pc_overlay, summary


# ── Interface Gradio ─────────────────────────────────────────────
with gr.Blocks(title="DefectVision", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 DefectVision — Industrial Defect Detection
    **YOLOv8** (supervised detection) + **PatchCore** (unsupervised anomaly detection)
    on MVTec AD dataset — bottle category
    """)

    with gr.Tabs():

        # ── Tab 1 : Combined ──────────────────────────────────────
        with gr.Tab("🔬 Analyse complète"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img_combined = gr.Image(type="pil", label="Image à analyser")
                    btn_combined       = gr.Button("🚀 Analyser", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        out_yolo_combined = gr.Image(label="YOLOv8 — Détections")
                        out_pc_combined   = gr.Image(label="PatchCore — Heatmap overlay")
                    out_summary = gr.Textbox(label="Résumé", lines=3)

            btn_combined.click(
                fn      = predict_combined,
                inputs  = [input_img_combined],
                outputs = [out_yolo_combined, out_pc_combined, out_summary],
            )

        # ── Tab 2 : YOLOv8 ───────────────────────────────────────
        with gr.Tab("📦 YOLOv8"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img_yolo = gr.Image(type="pil", label="Image à analyser")
                    btn_yolo       = gr.Button("🚀 Détecter", variant="primary")

                with gr.Column(scale=1):
                    out_yolo_img  = gr.Image(label="Résultat avec bounding boxes")
                    out_yolo_info = gr.Textbox(label="Détections", lines=6)

            btn_yolo.click(
                fn      = predict_yolov8,
                inputs  = [input_img_yolo],
                outputs = [out_yolo_img, out_yolo_info],
            )

        # ── Tab 3 : PatchCore ─────────────────────────────────────
        with gr.Tab("🧠 PatchCore"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img_pc = gr.Image(type="pil", label="Image à analyser")
                    btn_pc       = gr.Button("🚀 Analyser", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        out_pc_heatmap = gr.Image(label="Heatmap anomalie")
                        out_pc_overlay = gr.Image(label="Overlay")
                    out_pc_info = gr.Textbox(label="Score anomalie", lines=5)

            btn_pc.click(
                fn      = predict_patchcore,
                inputs  = [input_img_pc],
                outputs = [out_pc_heatmap, out_pc_overlay, out_pc_info],
            )

    gr.Markdown("""
    ---
    **Stack** : YOLOv8 · PatchCore · FastAPI · Gradio | **Dataset** : MVTec AD
    **GitHub** : [NaimMG/defect-vision](https://github.com/NaimMG/defect-vision)
    """)


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)