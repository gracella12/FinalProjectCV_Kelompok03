import random
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================
# 1. CONFIG
# ==========================================
VIDEO_PATH = "D:\\uni - 4th sem\\uni 4th\\CV\\FinalProjectCV_Kelompok03\\dataset\\Cv Pagi.mp4"
TARGET_W, TARGET_H = 1280, 720

ROI_NAMES  = ['traffic_light', 'detection_area', 'virtual_line']
ROI_COLORS = {
    'traffic_light':  (0,   0, 255),
    'detection_area': (0, 255,   0),
    'virtual_line':   (255,  0,   0),
}

HSV_RANGES = {
    'red': [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),
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

VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

# State tracking antar frame
crossed_ids = set()  # track_id yang sudah violation
prev_x1     = {}     # {track_id: x1} posisi ban belakang frame sebelumnya

def getColours(track_id):
    random.seed(track_id)
    return tuple(random.randint(0, 255) for _ in range(3))

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

def traffic_light_detection(frame, tl_pts):
    x, y, w, h = cv2.boundingRect(tl_pts)
    crop = frame[y:y+h, x:x+w]

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
    status = best if counts[best] > 30 else 'unknown'

    txt_color = STATUS_COLOR[status]
    cv2.putText(frame, f"Traffic Light: {status.upper()}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

    return status

def is_in_detection_area(x1, y2, da_pts):
    """
    Ban belakang = (x1, y2) pojok kiri bawah box.
    Kendaraan dari kiri→kanan, ban belakang di sisi kiri (x1).
    """
    return cv2.pointPolygonTest(da_pts, (float(x1), float(y2)), False) >= 0

def crosses_virtual_line(track_id, x1, sl_pts):
    """
    Violation hanya saat BARU melewati garis:
    prev_x1 < line_x  →  x1 >= line_x
    """
    line_x = int(np.mean([p[0] for p in sl_pts]))

    prev = prev_x1.get(track_id, None)
    prev_x1[track_id] = x1

    if prev is None:
        return False

    return (prev < line_x) and (x1 >= line_x)

def apply_roi(frame, tl_pts, da_pts):
    mask_tl = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_da = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_tl, [tl_pts], 255)
    cv2.fillPoly(mask_da, [da_pts], 255)
    masked_tl = cv2.bitwise_and(frame, frame, mask=mask_tl)
    masked_da = cv2.bitwise_and(frame, frame, mask=mask_da)
    return masked_tl, masked_da

def process_frame(frame, model, tracker, device_type, use_half,
                  tl_pts, da_pts, sl_pts, total_violations):

    # Step 1: Deteksi warna lampu
    tl_status = traffic_light_detection(frame, tl_pts)

    # Step 2: YOLO detect (bukan track) di detection area
    _, masked_da = apply_roi(frame, tl_pts, da_pts)
    results = model.predict(
        masked_da,
        imgsz=480,
        half=use_half,
        device=device_type,
        verbose=False
    )

    # Step 3: Siapkan deteksi untuk DeepSORT
    # Format yang dibutuhkan DeepSORT: list of ([x1,y1,w,h], conf, class_name)
    detections = []
    for result in results:
        class_names = result.names
        for box in result.boxes:
            conf       = float(box.conf[0])
            cls        = int(box.cls[0])
            class_name = class_names[cls]

            if conf > 0.4 and class_name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                # DeepSORT input: ([left, top, w, h], confidence, class_label)
                detections.append(([x1, y1, w, h], conf, class_name))

    # Step 4: Update DeepSORT tracker
    # tracks berisi objek dengan .track_id, .to_ltrb(), .det_class, .is_confirmed()
    tracks = tracker.update_tracks(detections, frame=masked_da)

    # Step 5: Gambar batas ROI
    cv2.polylines(frame, [da_pts], isClosed=True,  color=(0, 255,   0), thickness=2)
    cv2.polylines(frame, [tl_pts], isClosed=True,  color=(0,   0, 255), thickness=2)
    cv2.polylines(frame, [sl_pts], isClosed=False, color=(255,  0,   0), thickness=3)

    # Step 6: Loop tracks + cek pelanggaran
    for track in tracks:
        # Hanya proses track yang sudah dikonfirmasi DeepSORT
        # (minimal n_init frame berturut-turut terdeteksi)
        if not track.is_confirmed():
            continue

        track_id   = track.track_id
        class_name = track.det_class if track.det_class else 'unknown'
        colour     = getColours(track_id)

        # Ambil koordinat dari DeepSORT (format: left, top, right, bottom)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        in_area  = is_in_detection_area(x1, y2, da_pts)
        is_red   = tl_status == 'red'

        is_violation = False
        if track_id not in crossed_ids:
            just_crossed = crosses_virtual_line(track_id, x1, sl_pts)
            if in_area and just_crossed and is_red:
                crossed_ids.add(track_id)
                total_violations += 1
                is_violation = True

        # Gambar box
        if is_violation:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "VIOLATION",
                        (x1, max(y1 - 25, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label class + track ID
        cv2.putText(frame, f"{class_name} ID:{track_id}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        # Debug: titik ban belakang
        cv2.circle(frame, (x1, y2), 5, (0, 255, 255), -1)

    # Step 7: Counter pelanggaran
    cv2.putText(frame, f"Violations: {total_violations}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame, total_violations

if __name__ == '__main__':
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half    = device_type == 'cuda'
    print(f"Device: {device_type.upper()} | FP16: {use_half}")

    model = YOLO('yolo11n.pt')

    # Inisialisasi DeepSORT
    # max_age     : berapa frame track dipertahankan walau tidak terdeteksi
    # n_init      : berapa frame berturut-turut sebelum track dikonfirmasi
    # max_iou_distance: threshold IoU untuk matching
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        embedder='mobilenet',        # model re-ID ringan
        half=use_half,
        embedder_gpu=use_half,
    )

    # Ambil frame pertama untuk define ROI
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

    total_violations = 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    print("\nMemulai deteksi... tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        processed, total_violations = process_frame(
            frame, model, tracker, device_type, use_half,
            tl_pts, da_pts, sl_pts, total_violations
        )

        cv2.imshow("YOLO + DeepSORT", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Dihentikan oleh user.")
            break

    print(f"\nTotal pelanggaran terdeteksi: {total_violations}")
    cap.release()
    cv2.destroyAllWindows()