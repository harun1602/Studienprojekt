from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        lr0=0.001,
        device=0,        # CUDA
        workers=4,
        imgsz=640
    )

    metrics = model.val(device=0)
    print(metrics)

if __name__ == "__main__":
    freeze_support()   # 🔥 wichtig auf Windows
    main()
