from ultralytics import YOLO

# Veri seti klasör yapısı (sınıf başına klasör):
# fer2013/
#   train/ angry, disgust, fear, happy, neutral, sad, surprise
#   val/   angry, disgust, fear, happy, neutral, sad, surprise
VERI_YOLU = r"C:\Projeler\veri\fer2013"

if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt")  # daha iyi doğruluk için yolo11s-cls.pt
    model.train(
        data=VERI_YOLU,
        epochs=60,
        imgsz=128,
        batch=64,
        patience=15,
        project="runs/classify",
        name="duygu",
        fliplr=0.5,
        degrees=10,
        hsv_v=0.3,
    )
    # En iyi ağırlık: runs/classify/duygu/weights/best.pt
    metrics = model.val()
    print("Top-1 doğruluk:", metrics.top1)