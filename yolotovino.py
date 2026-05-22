from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Export ulang dengan mengunci resolusi di 480
print("Memulai re-export ke OpenVINO di resolusi 480...")
model.export(format='openvino', imgsz=480)
print("Export selesai!")