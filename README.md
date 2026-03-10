# 🔍 DefectVision — Industrial Defect Detection

> Computer Vision system for industrial quality control using **YOLOv8** (supervised detection) and **PatchCore** (unsupervised anomaly detection) on the **MVTec AD** dataset.

## 🏗️ Architecture
```
MVTec AD Dataset (bottle)
        │
        ├─── YOLOv8n (supervised) ────────────┐
        │    mAP50 = 0.09                      ├──▶ FastAPI ──▶ Gradio Demo
        └─── PatchCore (unsupervised) ─────────┘
             AUROC = 0.9976
```

## 🧠 Key Insight

> YOLOv8 (supervised) struggles with MVTec AD — only ~6 defect images available for training → mAP50=0.09.
> PatchCore (unsupervised) trains on **normal images only** and achieves AUROC=0.9976.
> This demonstrates why unsupervised anomaly detection is better suited for industrial inspection.

## 🧰 Stack

| Layer | Tools |
|-------|-------|
| Detection | YOLOv8n (Ultralytics) |
| Anomaly Detection | PatchCore from scratch (PyTorch) |
| Backbone | WideResNet50 (ImageNet) |
| API | FastAPI + Uvicorn |
| Demo | Gradio |
| Training | Google Colab T4 |
| Model Storage | HuggingFace Hub |

## 📁 Project Structure
```
defect-vision/
├── data/                   # Dataset documentation
├── models/                 # Model weights (gitignored)
├── notebooks/
│   ├── 01_explore_mvtec.ipynb       # Dataset exploration
│   ├── 02_yolov8_data_prep.ipynb    # YOLO format conversion
│   ├── 03_yolov8_training.ipynb     # YOLOv8 training
│   └── 04_patchcore.ipynb           # PatchCore from scratch
├── src/
│   ├── patchcore.py                 # PatchCore inference class
│   └── yolov8_inference.py          # YOLOv8 inference class
├── api/
│   └── main.py                      # FastAPI endpoints
├── app/
│   └── gradio_app.py                # Gradio interface
└── docker/
    └── Dockerfile
```

## 📊 Results

| Model | Approach | Metric | Value |
|-------|----------|--------|-------|
| YOLOv8n | Supervised | mAP50 | 0.0933 |
| YOLOv8n | Supervised | Recall | 0.8095 |
| PatchCore | Unsupervised | AUROC | **0.9976** |
| PatchCore | Unsupervised | F1 | **0.9920** |

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/NaimMG/defect-vision.git
cd defect-vision
pip install -r requirements.txt
```

### 2. Download models from HuggingFace
```python
from huggingface_hub import hf_hub_download
import os

os.makedirs("models/yolov8_bottle", exist_ok=True)
os.makedirs("models/patchcore_bottle", exist_ok=True)

hf_hub_download(
    repo_id="Chasston/defect-vision-yolov8-bottle",
    filename="best.pt",
    local_dir="models/yolov8_bottle"
)
hf_hub_download(
    repo_id="Chasston/defect-vision-patchcore-bottle",
    filename="memory_bank.pt",
    local_dir="models/patchcore_bottle"
)
```

### 3. Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start Gradio Demo
```bash
python app/gradio_app.py
```

### 5. Or use Docker
```bash
docker build -f docker/Dockerfile -t defect-vision .
docker run -p 8000:8000 -p 7860:7860 defect-vision
```

## 🤗 HuggingFace Models

- YOLOv8 : [Chasston/defect-vision-yolov8-bottle](https://huggingface.co/Chasston/defect-vision-yolov8-bottle)
- PatchCore : [Chasston/defect-vision-patchcore-bottle](https://huggingface.co/Chasston/defect-vision-patchcore-bottle)

## 👤 Author

**NaimMG** · [GitHub](https://github.com/NaimMG) · [HuggingFace](https://huggingface.co/Chasston)