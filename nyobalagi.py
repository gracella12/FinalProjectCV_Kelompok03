import cv2
import numpy as np
from ultralytics import YOLO

PATH = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Malam.mp4'
TARGET_W = 1280
TARGET_H = 720
ROI_NAMES = ['traffic_light', 'detection_area', 'line']
VEHICLES = ['car', 'motorcycle', 'bus', 'truck']
model = YOLO('yolo11n.pt')

HSV_RANGES = {
    'red': [
        (np.array([0, 130, 90]),   np.array([7,   255, 255])),
        (np.array([172, 130, 90]), np.array([180, 255, 255])),
    ],
    'yellow': [(np.array([10, 100, 80]), np.array([35, 255, 255]))],
    'green':  [(np.array([40,  70, 70]), np.array([90, 255, 255]))],
}
STATUS_COLORS = {
    'red':     (0,   0,   255),
    'yellow':  (0,   255, 255),
    'green':   (0,   255, 0),
    'unknown': (128, 128, 128),
}

prev_positions = {}  
crossed_ids    = set()

def segments_intersect(p1, p2, p3, p4):
    """
    True jika segmen p1-p2 berpotongan dengan segmen p3-p4.
    Menggunakan cross-product untuk mendeteksi perpotongan 2D.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Kasus collinear
    if d1 == 0 and on_segment(p3, p1, p4): return True
    if d2 == 0 and on_segment(p3, p2, p4): return True
    if d3 == 0 and on_segment(p1, p3, p2): return True
    if d4 == 0 and on_segment(p1, p4, p2): return True

    return False


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Gagal membaca frame dari '{video_path}'")
    return cv2.resize(frame, (TARGET_W, TARGET_H))


def define_roi(frame):
    points = []
    roi = {}
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


def apply_mask(frame, polygon_pts):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

def detect_traffic_light_state(frame, tl_pts):
    x, y, w, h = cv2.boundingRect(tl_pts)
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return 'unknown'

    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    counts = {}
    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(crop_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(crop_hsv, lower, upper)
        counts[color] = cv2.countNonZero(mask)

    max_color = max(counts, key=counts.get)
    status = max_color if counts[max_color] > 80 else 'unknown'

    cv2.putText(frame, f"Traffic Light: {status.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                STATUS_COLORS.get(status, (128, 128, 128)), 2)
    return status

def is_in_detection_area(point, da_pts):
    return cv2.pointPolygonTest(da_pts, (float(point[0]), float(point[1])), False) >= 0


def crosses_line(track_id, curr_x1, curr_y2, line_pts):
    """
    True jika jejak kendaraan (titik sebelumnya → titik sekarang)
    memotong virtual line (line_pts[0] → line_pts[-1]).
    Ini deteksi perpotongan segmen 2D yang benar untuk semua orientasi garis.
    """
    curr_pos = (curr_x1, curr_y2)
    prev_pos = prev_positions.get(track_id)
    prev_positions[track_id] = curr_pos   # simpan posisi sekarang untuk frame berikutnya

    if prev_pos is None:
        return False  # tidak ada riwayat posisi, belum bisa cek

    lp1 = tuple(line_pts[0])
    lp2 = tuple(line_pts[-1])

    return segments_intersect(prev_pos, curr_pos, lp1, lp2)


def check_violation(track_id, x1, y2, line_pts, da_pts, tl_status, total_violations):
    if track_id in crossed_ids:
        # Update posisi tetap dilakukan agar riwayat tidak basi
        prev_positions[track_id] = (x1, y2)
        return total_violations, False

    in_area     = is_in_detection_area((x1, y2), da_pts)
    just_crossed = crosses_line(track_id, x1, y2, line_pts)
    is_red      = tl_status == 'red'

    if in_area and just_crossed and is_red:
        crossed_ids.add(track_id)
        total_violations += 1
        return total_violations, True

    return total_violations, False

def detect_vehicles(frame, masked_da, da_pts, line_pts, tl_status, total_violations):
    results = model.track(masked_da, verbose=False, imgsz=480, persist=True)

    for result in results:
        for box in result.boxes:
            conf       = float(box.conf[0])
            cls        = int(box.cls[0])
            class_name = result.names[cls]

            if conf < 0.4 or class_name not in VEHICLES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if track_id is not None:
                total_violations, _ = check_violation(
                    track_id, x1, y2,
                    line_pts, da_pts, tl_status, total_violations
                )

            if track_id in crossed_ids:
                box_color = (0, 0, 255)
                label     = f"VIOLATION {class_name} {conf:.2f}"
            else:
                box_color = (0, 255, 0)
                label     = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            # Titik referensi bawah kendaraan
            cv2.circle(frame, (x1, y2), 5, (0, 255, 255), -1)

    return frame, total_violations

def read_video_with_roi(roi):
    cap = cv2.VideoCapture(PATH)
    tl_pts   = np.array(roi['traffic_light'],   dtype=np.int32)
    da_pts   = np.array(roi['detection_area'],  dtype=np.int32)
    line_pts = np.array(roi['line'],            dtype=np.int32)
    total_violations = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        tl_status = detect_traffic_light_state(frame, tl_pts)
        masked_da = apply_mask(frame, da_pts)

        frame, total_violations = detect_vehicles(
            frame, masked_da, da_pts, line_pts, tl_status, total_violations
        )

        cv2.putText(frame, f"Total Violations: {total_violations}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Gambar ROI
        cv2.polylines(frame, [tl_pts],   isClosed=True,  color=(0, 0, 255),   thickness=2)
        cv2.polylines(frame, [da_pts],   isClosed=True,  color=(0, 255, 0),   thickness=2)
        cv2.polylines(frame, [line_pts], isClosed=False, color=(255, 0, 0),   thickness=3)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Dihentikan user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    first_frame = get_first_frame(PATH)
    print("Definisikan ROI (traffic_light → detection_area → line)...")
    roi = define_roi(first_frame)
    print(f"\nROI selesai: {list(roi.keys())}")
    print("\nMemulai deteksi... tekan 'q' untuk keluar.")
    read_video_with_roi(roi)