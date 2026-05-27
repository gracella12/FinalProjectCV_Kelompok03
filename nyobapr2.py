import cv2
import numpy as np
import os
from ultralytics import YOLO

#0. Read file, setup, and preprocessing
#1. Fokus di traffic light detection
#2. Deteksi kendaraan di detection area using YOLOv11

PATH = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Malam.mp4'

ROI_NAMES = ['traffic_light', 'detection_area', 'line']

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

VEHICLES = {2, 3, 5, 7}
model = YOLO('yolo11n.pt')

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Gagal membaca frame dari '{video_path}'")
    return cv2.resize(frame, (1280, 720))

def define_roi(frame):
    points = []
    roi = {}
    display_img = frame.copy()

    def click_event(event, x, y, flags, params):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 4, (0, 255, 255), -1)
            if len(points) >= 2:
                cv2.line(display_img, points[-2], points[-1], (0, 255, 255), 1)
            cv2.imshow("Define ROI", display_img)

    cv2.namedWindow("Define ROI")
    cv2.setMouseCallback("Define ROI", click_event)

    BOX_ROIS = {'traffic_light', 'detection_area'}  # tambah nama lain di sini kalau perlu

    for name in ROI_NAMES:
        points.clear()
        display_img = frame.copy()
        for saved_name, saved_pts in roi.items():
            pts = np.array(saved_pts, dtype=np.int32)
            cv2.polylines(display_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(display_img, saved_name, tuple(saved_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if name in BOX_ROIS:
            cv2.putText(display_img, f"ROI: {name}  |  drag kotak  |  Enter = konfirmasi",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
            cv2.imshow("Define ROI", display_img)
            x, y, w, h = cv2.selectROI("Define ROI", display_img, False, False)
            box_pts = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            roi[name] = box_pts
            cv2.setMouseCallback("Define ROI", click_event)  # kembalikan callback
            
        else:
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

def draw_roi_overlay(frame, roi):
    for name, pts_list in roi.items():
        if name != 'line':
            continue  # skip semua selain line
        pts = np.array(pts_list, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

def detect_traffic_light_state(frame, tl_box_pts):
    pts = np.array(tl_box_pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)  
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

def vehicles_detected(frame, da_pts):
    results = model.predict(frame, verbose=False, imgsz=640)

    if results[0].boxes is None:
        return frame, 0

    vehicle_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])

        if cls_id not in VEHICLES:
            continue

        #mencari koordinat bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        #center point kendaraan
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if not is_in_detection_area(center, da_pts):
            continue

        vehicle_count += 1
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, center, 4, (0, 255, 255), -1)

    return frame, vehicle_count

#MAIN PART
def read_video_with_roi(roi):
    cap = cv2.VideoCapture(PATH)
   

    if not cap.isOpened():
        raise ValueError(f"Gagal membuka video: {PATH}")

    da_pts = np.array(roi['detection_area'], dtype=np.int32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))

        draw_roi_overlay(frame, roi)

        # Deteksi traffic light
        if 'traffic_light' in roi:
            status = detect_traffic_light_state(frame, roi['traffic_light'])

        # detection area
        if 'detection_area' in roi:
            frame, _ = vehicles_detected(frame, da_pts)

        cv2.imshow("Output", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    first_frame = get_first_frame(PATH)
    roi = define_roi(first_frame)
    print(f"\nROI selesai: {list(roi.keys())}")

    read_video_with_roi(roi)
