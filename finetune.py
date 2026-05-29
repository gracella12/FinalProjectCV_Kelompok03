from roboflow import Roboflow
rf = Roboflow(api_key="ceILv1OhczfWqWsZynZb")
project = rf.workspace("gracellas-workspace-1yzyy").project("projekcv-pjkk5")
version = project.version(1)
dataset = version.download("yolov11")
                
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data = dataset.location + "/data.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 16,
    workers = 4,
    device = "cpu",
    patience = 10,
)