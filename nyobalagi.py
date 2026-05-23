import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
PATH      = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Pagi.mp4'
TARGET_W  = 1280
TARGET_H  = 720
ROI_NAMES = ['traffic_light', 'detection_area', 'line']  # list, bukan set
VEHICLES  = ['car', 'motorcycle', 'bus', 'truck']

model = YOLO('yolo11n.pt')

HSV_RANGES = {
    'red': [
        (np.array([0, 150, 120]), np.array([5, 255, 255])),
        (np.array([175, 150, 120]), np.array([180, 255, 255])),  # merah wrap-around
    ],
    'yellow': [
        (np.array([10, 100, 80]), np.array([35, 255, 255])), 
    ],
    'green': [
        (np.array([40,  70,  70]),   np.array([90,  255, 255])),
    ],
}

STATUS_COLORS = {
    'red':     (0,   0,   255),
    'yellow':  (0,   255, 255),
    'green':   (0,   255,   0),
    'unknown': (128, 128, 128),
}

# ==========================================
# AMBIL FRAME PERTAMA
# ==========================================
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Gagal membaca frame dari '{video_path}'")
    return cv2.resize(frame, (TARGET_W, TARGET_H))

# ==========================================
# DEFINE ROI
# ==========================================
def define_roi(frame):
    points      = []
    roi         = {}
    display_img = frame.copy()

    def click_event(event, x, y, flags, params):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 5, (0, 255, 255), -1)
            if len(points) >= 2:
                cv2.line(display_img, points[-2], points[-1], (0, 255, 255), 1)
            cv2.imshow("Define ROI", display_img)

    cv2.namedWindow("Define ROI")
    cv2.setMouseCallback("Define ROI", click_event)

    for name in ROI_NAMES:
        points.clear()
        display_img = frame.copy()

        for saved_name, saved_pts in roi.items():
            pts = np.array(saved_pts, dtype=np.int32)
            cv2.polylines(display_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(display_img, saved_name, tuple(saved_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(display_img,
                    f"ROI: {name}  |  klik titik  |  's' = simpan & lanjut",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
        print(f"\n[{name}] Klik titik-titik ROI, tekan 's' jika sudah selesai.")
        cv2.imshow("Define ROI", display_img)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('s') and len(points) >= 2:
                break

        cv2.line(display_img, points[-1], points[0], (0, 255, 255), 1)
        cv2.imshow("Define ROI", display_img)
        cv2.waitKey(400)

        roi[name] = points.copy()
        print(f"  ✓ '{name}' tersimpan: {len(points)} titik")

    cv2.destroyWindow("Define ROI")
    return roi

# ==========================================
# APPLY MASK
# ==========================================
def apply_mask(frame, polygon_pts):
    """Hitamkan semua area di luar polygon, kirim ke YOLO"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

# ==========================================
# DETEKSI KENDARAAN
# ==========================================
def detect_vehicles(frame, masked_da):
    """
    YOLO predict di masked_da (area detection saja).
    Gambar box hasil deteksi ke frame asli.
    """
    results = model.predict(masked_da, verbose=False, imgsz=480)

    for result in results:
        for box in result.boxes:
            conf       = float(box.conf[0])
            cls        = int(box.cls[0])
            class_name = result.names[cls]

            if conf > 0.4 and class_name in VEHICLES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Box kendaraan
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label class + confidence
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Titik ban belakang (pojok kiri bawah — kendaraan dari kiri ke kanan)
                cv2.circle(frame, (x1, y2), 5, (0, 255, 255), -1)

    return frame

def crop_traffic_light(frame, tl_pts):
    x, y, w, h = cv2.boundingRect(tl_pts)
    crop = frame[y:y+h, x:x+w]
    return crop

def detect_traffic_light_state(frame, tl_pts):
    crop = crop_traffic_light(frame, tl_pts)

    if crop.size == 0:
        return 'unknown'
    
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    counts = {}

    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(crop_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(crop_hsv, lower, upper) #bitwise OR untuk gabungkan mask warna yang sama (misal merah wrap-around)
        counts[color] = cv2.countNonZero(mask)


    max_color = max(counts, key=counts.get)
    status = max_color if counts[max_color] > 30 else 'unknown'

    text_color = STATUS_COLORS[status]
    cv2.putText(frame, f"Traffic Light: {status.upper()}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    return status

# LOOP UTAMA
def read_video_with_roi(roi):
    cap = cv2.VideoCapture(PATH)

    tl_pts   = np.array(roi['traffic_light'],  dtype=np.int32)
    da_pts   = np.array(roi['detection_area'], dtype=np.int32)
    line_pts = np.array(roi['line'],           dtype=np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        tl_status = detect_traffic_light_state(frame, tl_pts)
        # 1. Mask detection area → kirim ke YOLO
        masked_da = apply_mask(frame, da_pts)

        # 2. Deteksi kendaraan di masked area, gambar box ke frame asli
        frame = detect_vehicles(frame, masked_da)

        # 3. Gambar batas ROI di atas frame
        cv2.polylines(frame, [tl_pts],   isClosed=True,  color=(0,   0,   255), thickness=2)
        cv2.polylines(frame, [da_pts],   isClosed=True,  color=(0,   255,   0), thickness=2)
        cv2.polylines(frame, [line_pts], isClosed=False, color=(255,   0,   0), thickness=3)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Dihentikan user.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Step 1: ambil frame pertama
    first_frame = get_first_frame(PATH)

    # Step 2: define ROI
    print("Definisikan ROI (traffic_light → detection_area → line)...")
    roi = define_roi(first_frame)
    print(f"\nROI selesai: {list(roi.keys())}")

    # Step 3: jalankan deteksi
    print("\nMemulai deteksi... tekan 'q' untuk keluar.")
    read_video_with_roi(roi)