import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path


class YOLOv8Inference:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def predict(self, image: Image.Image):
        results = self.model.predict(
            source  = image,
            conf    = self.conf_threshold,
            verbose = False,
        )
        result = results[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "x1"        : round(x1, 2),
                "y1"        : round(y1, 2),
                "x2"        : round(x2, 2),
                "y2"        : round(y2, 2),
                "confidence": round(float(box.conf[0]), 4),
                "class"     : int(box.cls[0]),
                "label"     : result.names[int(box.cls[0])],
            })

        annotated = Image.fromarray(result.plot())

        return {
            "n_detections": len(detections),
            "detections"  : detections,
            "annotated_image": annotated,
        }