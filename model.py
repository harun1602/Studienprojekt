from ultralytics import YOLO


model = YOLO("yolov8s.pt")


model.train(
    data="data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    lr0=0.001,
    device="cpu",
    workers=4
)


metrics = model.val()