from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    metadata_path = Path("data", "african_wildlife", "processed", "dataset.yaml")
    assert metadata_path.exists()

    results = model.train(data=metadata_path, epochs=10, imgsz=256)
