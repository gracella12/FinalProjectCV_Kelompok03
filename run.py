import cv2
import json
import numpy as np
import torch
import random
from ultralytics import YOLO

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

with open("roi_config.json") as f:
    roi_config = json.load(f)

tl_points = np.array(roi_config["traffic_light"], dtype=np.int32)
da_points = np.array(roi_config["detection_area"], dtype=np.int32)
sl_points = np.array(roi_config["virtual_line"], dtype=np.int32)

def apply_roi(frame):
    mask_tl = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_da = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask_tl, [tl_points], 255)
    cv2.fillPoly(mask_da, [da_points], 255)

    masked_frame_tl = cv2.bitwise_and(frame, frame, mask=mask_tl)
    masked_frame_da = cv2.bitwise_and(frame, frame, mask=mask_da)
    
    return masked_frame_tl, masked_frame_da

# ==========================================
# 4. FUNGSI PEMROSESAN PER FRAME
# ==========================================
def process_frame(frame, model, device_type, use_half):
    masked_tl, masked_da = apply_roi(frame)
    
    results = model.track(
        masked_da, 
        stream=True,
        imgsz=480,       
        half=use_half,   
        device=device_type,
        verbose=False     
    )
    
    # Visualisasi batas ROI di frame asli
    cv2.polylines(frame, [da_points], isClosed=True, color=(0, 255, 0), thickness=2)      
    cv2.polylines(frame, [tl_points], isClosed=True, color=(0, 0, 255), thickness=2)      
    cv2.polylines(frame, [sl_points], isClosed=False, color=(255, 0, 0), thickness=3)     

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                colour = getColours(cls)

                if class_name in ['car', 'motorcycle', 'bus', 'truck']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)
                    
    return frame

# ==========================================
# 5. BLOK UTAMA PEMBACAAN VIDEO
# ==========================================
if __name__ == '__main__':
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half = True if device_type == 'cuda' else False
    print(f"Menggunakan Device: {device_type.upper()} | FP16 Half Precision: {use_half}")

    model = YOLO('yolo11n.pt') 

    video_path = "D:\\uni - 4th sem\\uni 4th\\CV\\FinalProjectCV_Kelompok03\\dataset\\Cv Pagi.mp4"
    videoCap = cv2.VideoCapture(video_path)

    print("Memulai preview video... Tekan 'q' pada keyboard untuk keluar.")

    # LOOPING PEMBACAAN VIDEO
    while True:
        ret, frame = videoCap.read()
        if not ret:
            print("\nSelesai! Video berakhir.")
            break
        
        frame = cv2.resize(frame, (1280, 720)) # Resize untuk konsistensi tampilan
        processed_frame = process_frame(frame, model, device_type, use_half)
        
        # Tampilkan frame yang sudah diproses ke layar
        cv2.imshow('Preview YOLO ROI', processed_frame)
        
        # Tunggu 1 milidetik, dan jika tombol 'q' ditekan, hentikan loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nPreview dihentikan oleh user.")
            break

    videoCap.release()
    cv2.destroyAllWindows()

    