import cv2
import random
import torch
from ultralytics import YOLO

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# Define ROI

# Mengecek apakah GPU (CUDA) tersedia. Jika tidak, akan otomatis pakai CPU.
# Ini mencegah error saat menggunakan fitur half-precision (FP16).
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
use_half = True if device_type == 'cuda' else False

# Inisialisasi model
model = YOLO('yolo11n.pt')

video_path = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Siang.mp4'
videoCap = cv2.VideoCapture(video_path)

# Mengatur ukuran window agar pas di layar
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 1280, 720)

while True:
    ret, frame = videoCap.read()
    if not ret:
        break
    
    # --- BAGIAN YANG DIOPTIMASI ---
    # Menggunakan predict (lebih ringan dari track jika tidak butuh ID)
    # imgsz diaktifkan ke 480, half precision dan device disesuaikan otomatis
    results = model.predict(
        frame, 
        stream=True, 
        imgsz=480, # paksa model untuk resize input ke 480x480, lebih cepat dari default 640
        half=use_half, # otomatis pakai FP16 jika GPU tersedia
        device=device_type,
        verbose=True # tampilin log di terminal atau tidak
    )

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                colour = getColours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

    cv2.imshow('Frame', frame)
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()