# 📦 Data

## Dataset : MVTec AD

Le dataset **MVTec Anomaly Detection** est un benchmark de référence pour la détection de défauts industriels.

- **15 catégories** : bouteilles, câbles, capsules, tapis, noisettes, métal, pilules, vis, tuiles, bois...
- **5000+ images** haute résolution
- **Anomalies annotées** avec masques de segmentation

## Accès

Le dataset est chargé directement depuis HuggingFace Hub dans les notebooks Colab.  
Aucun téléchargement manuel requis.

> ⚠️ Accepter la licence MVTec AD sur HuggingFace avant d'utiliser le dataset :  
> https://huggingface.co/datasets/mvtec/mvtec_ad

## Structure du dataset
```
mvtec_ad/
├── bottle/
│   ├── train/good/        # Images sans défaut (entraînement)
│   └── test/
│       ├── good/          # Images sans défaut (test)
│       └── broken_large/  # Images avec défauts (test)
├── cable/
├── capsule/
└── ...
```

## Utilisation dans les notebooks
```python
from datasets import load_dataset

ds = load_dataset("mvtec/mvtec_ad", "bottle")
```