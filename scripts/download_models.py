"""
Script pour télécharger les modèles depuis HuggingFace Hub.
Usage : python scripts/download_models.py
"""
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

MODELS = [
    {
        "repo_id"  : "Chasston/defect-vision-yolov8-bottle",
        "filename" : "best.pt",
        "local_dir": "models/yolov8_bottle",
    },
    {
        "repo_id"  : "Chasston/defect-vision-patchcore-bottle",
        "filename" : "memory_bank.pt",
        "local_dir": "models/patchcore_bottle",
    },
]

if __name__ == "__main__":
    for m in MODELS:
        dest = Path(m["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / m["filename"]

        if target.exists():
            print(f"✅ Déjà présent : {target}")
            continue

        print(f"📦 Téléchargement : {m['repo_id']} / {m['filename']}")
        hf_hub_download(
            repo_id   = m["repo_id"],
            filename  = m["filename"],
            local_dir = str(dest),
        )
        print(f"✅ Sauvegardé : {target}")

    print("\n🎉 Tous les modèles sont prêts !")