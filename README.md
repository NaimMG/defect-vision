# 🔍 DefectVision — Industrial Defect Detection

> Computer Vision system for industrial quality control using **YOLOv8** (supervised detection) and **PatchCore** (unsupervised anomaly detection) on the **MVTec AD** dataset.

## 🏗️ Architecture
```
MVTec AD Dataset
      │
      ├─── YOLOv8 (supervised) ──────────┐
      │                                   ├──▶ FastAPI ──▶ Gradio Demo
      └─── PatchCore (unsupervised) ──────┘
```

## 🧰 Stack

| Layer | Tools |
|-------|-------|
| Detection | YOLOv8, PatchCore |
| API | FastAPI |
| Demo | Gradio |
| Infra | Docker, HuggingFace Hub |
| Training | Google Colab (T4 GPU) |

## 📁 Project Structure
```
defect-vision/
├── data/           # Dataset scripts & samples
├── models/         # Model definitions
├── notebooks/      # Colab training notebooks
├── src/            # Core logic (preprocessing, inference)
├── api/            # FastAPI app
├── app/            # Gradio interface
└── docker/         # Dockerfiles
```

## 🚀 Quickstart
```bash
# Coming soon
```

## 📊 Results

*To be updated after training.*

## 👤 Author

**NaimMG** · [GitHub](https://github.com/NaimMG) · [HuggingFace](https://huggingface.co/Chasston)