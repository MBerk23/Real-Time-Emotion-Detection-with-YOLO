# YOLO ile Yüz İfadesinden Duygu Analizi

Ultralytics YOLO sınıflandırma modeliyle (`yolo11n-cls`) eğitilen, yüz ifadesinden duyguyu ve **güvenilirlik oranını** gösteren basit bir masaüstü uygulaması.

- `train.py`: Veri setiyle YOLO sınıflandırma modelini eğitir.
- `app.py`: Tkinter arayüzü. Kameradan canlı veya seçilen resimden duygu tahmini yapar, sonucu yüzde olarak yazar.

## Nasıl Çalışır?

1. OpenCV Haar cascade ile görüntüdeki yüz bulunur.
2. Yüz kırpılır ve eğitilmiş YOLO modeline verilir.
3. Modelin en yüksek olasılıklı sınıfı **duygu**, bu sınıfın olasılığı **güvenilirlik oranı** olarak gösterilir.

Örnek çıktı: `Duygu: Mutlu   Güvenilirlik: %87.3`

## Sınıflar

`angry` (Kızgın), `disgust` (İğrenme), `fear` (Korku), `happy` (Mutlu), `neutral` (Nötr), `sad` (Üzgün), `surprise` (Şaşkın)

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/duygu-yolo.git
cd duygu-yolo
pip install -r requirements.txt
```

`requirements.txt` içeriği:

```
ultralytics
opencv-python
pillow
```

## Veri Seti

Veri seti repoya dahil değildir. [FER2013](https://www.kaggle.com/datasets/msambare/fer2013), RAF-DB veya AffectNet kullanılabilir. Klasör yapısı sınıf başına bir klasör olmalıdır:

```
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── val/
    └── (aynı sınıf klasörleri)
```

## Eğitim

`train.py` içindeki `VERI_YOLU` değişkenini veri setinin yoluna göre düzenleyin, ardından:

```bash
python train.py
```

En iyi ağırlık `runs/classify/duygu/weights/best.pt` altına kaydedilir.

## Arayüzü Çalıştırma

```bash
python app.py
```

- **Kamerayı Başlat**: Canlı görüntüde yüzleri bulur, duygu ve güven oranını yazar.
- **Resim Aç**: Seçilen resim üzerinde tahmin yapar.
- **Durdur**: Kamerayı kapatır.

Model yolu `app.py` içindeki `MODEL_YOLU` değişkeniyle, görüntü boyutu `IMGSZ` ile ayarlanır. `IMGSZ` değeri `train.py` içindeki `imgsz` ile aynı olmalıdır.

## Notlar

- FER2013 etiketleri gürültülüdür, doğruluk genelde %65-70 civarında kalır.
- `disgust` sınıfı veri setinde çok az örneğe sahiptir.
- Işık, yüz açısı ve görüntü kalitesi güvenilirlik oranını belirgin şekilde etkiler.
- Düşük güvenli tahminlerde (örneğin %50 altı) sonuç dikkatle yorumlanmalıdır.

