import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x  = self.layer0(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


class PatchCoreInference:
    def __init__(self, memory_bank_path: str, device: str = "cpu"):
        self.device   = torch.device(device)
        self.upsample = nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False)
        self.extractor = FeatureExtractor().to(self.device).eval()

        data = torch.load(memory_bank_path, map_location="cpu", weights_only=False)
        memory_bank  = data["memory_bank"]
        self.threshold = float(data.get("best_thresh", 4.5))

        self.knn = NearestNeighbors(n_neighbors=9, metric="euclidean",
                                     algorithm="ball_tree", n_jobs=-1)
        self.knn.fit(memory_bank.numpy())

    def predict(self, image: Image.Image):
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f2, f3   = self.extractor(tensor)
            f3_up    = self.upsample(f3)
            combined = torch.cat([f2, f3_up], dim=1)
            B, C, H, W = combined.shape
            patches  = combined.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

        distances, _ = self.knn.kneighbors(patches, n_neighbors=1)
        distances    = distances.squeeze()
        heatmap      = distances.reshape(28, 28)
        score        = float(distances.max())
        is_defect    = score >= self.threshold

        return {
            "score"     : round(score, 4),
            "threshold" : round(self.threshold, 4),
            "is_defect" : is_defect,
            "heatmap"   : heatmap.tolist(),
        }