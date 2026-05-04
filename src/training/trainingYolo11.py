from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATASET_DIR = SCRIPT_DIR / "football-players-detection-1"

    rf = Roboflow(api_key="ZAdNbwMa0PU55HdiDvK4")
    project = rf.workspace("leos-workspace-rediw").project(
        "football-players-detection-3zvbc-uefnz"
    )
    version = project.version(1)
    dataset = version.download("yolov11", location=DATASET_DIR)

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16
    )

if __name__ == "__main__":
    main()