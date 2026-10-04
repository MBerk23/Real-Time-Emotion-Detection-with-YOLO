import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_YOLU = r"C:\Projeler\duygu\models\duygu.pt"
IMGSZ = 128  # train.py ile aynı olmalı

TR = {
    "angry": "Kızgın", "disgust": "İğrenme", "fear": "Korku",
    "happy": "Mutlu", "neutral": "Nötr", "sad": "Üzgün", "surprise": "Şaşkın",
}

model = YOLO(MODEL_YOLU)
yuz_bulucu = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def siniflandir(goruntu):
    r = model.predict(goruntu, imgsz=IMGSZ, verbose=False)[0]
    ad = r.names[r.probs.top1]
    return ad, float(r.probs.top1conf)


def analiz_et(kare, yuz_yoksa_tum_kare=False):
    """Karedeki yüzleri bulur, üzerine çizer, [(duygu, güven), ...] döndürür."""
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_bulucu.detectMultiScale(gri, 1.2, 5, minSize=(60, 60))
    sonuclar = []
    for (x, y, w, h) in yuzler:
        ad, guven = siniflandir(kare[y:y + h, x:x + w])
        sonuclar.append((ad, guven))
        cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # OpenCV Türkçe karakter çizemez, kare üzerinde İngilizce etiket
        cv2.putText(kare, f"{ad} %{guven * 100:.0f}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if not sonuclar and yuz_yoksa_tum_kare:
        sonuclar.append(siniflandir(kare))
    return sonuclar


class Uygulama:
    def __init__(self, pencere):
        self.pencere = pencere
        pencere.title("Duygu Analizi (YOLO)")
        self.cap = None
        self.calisiyor = False

        self.ekran = tk.Label(pencere, width=80, height=30, bg="black")
        self.ekran.pack(padx=10, pady=10)

        self.sonuc = tk.Label(pencere, text="Duygu: -   Güvenilirlik: -",
                              font=("Segoe UI", 16, "bold"))
        self.sonuc.pack(pady=5)

        cerceve = tk.Frame(pencere)
        cerceve.pack(pady=10)
        tk.Button(cerceve, text="Kamerayı Başlat", width=16,
                  command=self.kamera_baslat).grid(row=0, column=0, padx=5)
        tk.Button(cerceve, text="Durdur", width=16,
                  command=self.durdur).grid(row=0, column=1, padx=5)
        tk.Button(cerceve, text="Resim Aç", width=16,
                  command=self.resim_ac).grid(row=0, column=2, padx=5)

        pencere.protocol("WM_DELETE_WINDOW", self.kapat)

    def goster(self, kare):
        rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, int(640 * rgb.shape[0] / rgb.shape[1])))
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.ekran.configure(image=img, width=img.width(), height=img.height())
        self.ekran.image = img

    def yaz(self, sonuclar):
        if not sonuclar:
            self.sonuc.config(text="Yüz bulunamadı")
            return
        ad, guven = max(sonuclar, key=lambda s: s[1])
        self.sonuc.config(
            text=f"Duygu: {TR.get(ad, ad)}   Güvenilirlik: %{guven * 100:.1f}")

    def kamera_baslat(self):
        if self.calisiyor:
            return
        self.cap = cv2.VideoCapture(0)
        self.calisiyor = True
        self.dongu()

    def dongu(self):
        if not self.calisiyor:
            return
        ok, kare = self.cap.read()
        if ok:
            kare = cv2.flip(kare, 1)
            self.yaz(analiz_et(kare))
            self.goster(kare)
        self.pencere.after(30, self.dongu)

    def durdur(self):
        self.calisiyor = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def resim_ac(self):
        yol = filedialog.askopenfilename(
            filetypes=[("Resim", "*.jpg *.jpeg *.png *.bmp")])
        if not yol:
            return
        self.durdur()
        kare = cv2.imread(yol)
        if kare is None:
            self.sonuc.config(text="Resim okunamadı")
            return
        self.yaz(analiz_et(kare, yuz_yoksa_tum_kare=True))
        self.goster(kare)

    def kapat(self):
        self.durdur()
        self.pencere.destroy()


if __name__ == "__main__":
    kok = tk.Tk()
    Uygulama(kok)
    kok.mainloop()