import random
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# 1. CONFIG
VIDEO_PATH = "D:\\uni - 4th sem\\uni 4th\\CV\\FinalProjectCV_Kelompok03\\dataset\\Cv Malam.mp4"
TARGET_W, TARGET_H = 1280, 720

ROI_NAMES  = ['traffic_light', 'detection_area', 'virtual_line']
ROI_COLORS = {
    'traffic_light':  (0,   0, 255),
    'detection_area': (0, 255,   0),
    'virtual_line':   (255,  0,   0),
}

# HSV ranges yang benar
HSV_RANGES = {
    'red': [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),  # merah wrap-around
    ],
    'yellow': [
        (np.array([20, 100, 100]),   np.array([35,  255, 255])),
    ],
    'green': [
        (np.array([40,  70,  70]),   np.array([90,  255, 255])),
    ],
}

STATUS_COLOR = {
    'red':     (0,   0,   255),
    'yellow':  (0,   255, 255),
    'green':   (0,   255,   0),
    'unknown': (128, 128, 128),
}

# 2. FUNGSI WARNA BOX KENDARAAN
def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# 3. DEFINE ROI DARI FRAME PERTAMA
def define_roi(frame_ref):
    all_rois    = {}
    display_img = frame_ref.copy()
    current_pts = []

    def click_event(event, x, y, flags, params):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append((x, y))
            cv2.circle(display_img, (x, y), 5, (0, 255, 255), -1)
            if len(current_pts) >= 2:
                cv2.line(display_img, current_pts[-2], current_pts[-1], (0, 255, 255), 2)
            cv2.imshow(win, display_img)

    win = "Define ROI  |  klik titik  |  'q' = selesai ROI ini"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, click_event)

    for roi_name in ROI_NAMES:
        current_pts.clear()
        display_img = frame_ref.copy()

        for name, pts in all_rois.items():
            color = ROI_COLORS[name]
            cv2.polylines(display_img, [np.array(pts, dtype=np.int32)],
                          isClosed=True, color=color, thickness=2)
            cv2.putText(display_img, name, tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(display_img, f"Klik titik untuk: {roi_name}  |  'q' selesai",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"\n  [{roi_name}] Klik titik-titik ROI, tekan 'q' jika sudah selesai.")
        cv2.imshow(win, display_img)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q') and len(current_pts) >= 2:
                break

        cv2.line(display_img, current_pts[-1], current_pts[0], (0, 255, 255), 2)
        cv2.imshow(win, display_img)
        cv2.waitKey(300)

        all_rois[roi_name] = list(current_pts)
        print(f"    ✓ {len(current_pts)} titik tersimpan untuk '{roi_name}'")

    cv2.destroyWindow(win)
    return all_rois

# 4. CROP TRAFFIC LIGHT DARI POLYGON
def crop_traffic_light(frame, tl_pts):
    """Crop bounding box dari polygon tl_pts"""
    x, y, w, h = cv2.boundingRect(tl_pts)
    return frame[y:y+h, x:x+w], (x, y)

# 5. DETEKSI WARNA LAMPU (HSV)
def traffic_light_detection(frame, tl_pts):
    crop, (ox, oy) = crop_traffic_light(frame, tl_pts)
    if crop.size == 0:
        return 'unknown'

    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    counts = {}

    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        counts[color] = cv2.countNonZero(mask)

    best   = max(counts, key=counts.get)
    status = best if counts[best] > 80 else 'unknown'

    # Tampilkan status di frame
    label     = f"Traffic Light: {status.upper()}"
    txt_color = STATUS_COLOR[status]
    cv2.putText(frame, label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

    return status

# Apply ROI
def apply_roi(frame, tl_pts, da_pts):
    mask_tl = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_da = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_tl, [tl_pts], 255)
    cv2.fillPoly(mask_da, [da_pts], 255)
    masked_tl = cv2.bitwise_and(frame, frame, mask=mask_tl)
    masked_da = cv2.bitwise_and(frame, frame, mask=mask_da)
    return masked_tl, masked_da

def process_frame(frame, model, device_type, use_half, tl_pts, da_pts, sl_pts):
    # Deteksi warna lampu dari crop ROI traffic light
    tl_status = traffic_light_detection(frame, tl_pts)

    # YOLO hanya di detection area
    _, masked_da = apply_roi(frame, tl_pts, da_pts)
    results = model.track(
        masked_da,
        stream=True,
        imgsz=480,
        half=use_half,
        device=device_type,
        verbose=False
    )

    # Gambar batas ROI
    cv2.polylines(frame, [da_pts], isClosed=True,  color=(0, 255,   0), thickness=2)
    cv2.polylines(frame, [tl_pts], isClosed=True,  color=(0,   0, 255), thickness=2)
    cv2.polylines(frame, [sl_pts], isClosed=False, color=(255,  0,   0), thickness=3)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls        = int(box.cls[0])
                class_name = class_names[cls]
                conf       = float(box.conf[0])
                colour     = getColours(cls)
                if class_name in ['car', 'motorcycle', 'bus', 'truck']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return frame

if __name__ == '__main__':
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half    = device_type == 'cuda'
    print(f"Device: {device_type.upper()} | FP16: {use_half}")

    model = YOLO('yolo11n.pt')

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Gagal membaca video!")
        exit()

    first_frame = cv2.resize(first_frame, (TARGET_W, TARGET_H))

    print("\nDefine ROI dulu sebelum deteksi dimulai...")
    rois   = define_roi(first_frame)
    tl_pts = np.array(rois["traffic_light"],  dtype=np.int32)
    da_pts = np.array(rois["detection_area"], dtype=np.int32)
    sl_pts = np.array(rois["virtual_line"],   dtype=np.int32)

    cap = cv2.VideoCapture(VIDEO_PATH)
    print("\nMemulai deteksi... tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        frame     = cv2.resize(frame, (TARGET_W, TARGET_H))
        processed = process_frame(frame, model, device_type, use_half,
                                  tl_pts, da_pts, sl_pts)

        cv2.imshow("YOLO Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Dihentikan oleh user.")
            break

    cap.release()
    cv2.destroyAllWindows()