import cv2
import numpy as np
import os
from streamlit import status
from ultralytics import YOLO

#0. Read file, setup, and preprocessing
#1. Fokus di traffic light detection
#2. Deteksi kendaraan di detection area using YOLOv8
#3. Buat logic untuk menentukan violation

PATH = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Pagi.mp4'

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
CROSS_DIST = 15  # jarak maksimal titik ke garis untuk dianggap crossing

violated_ids = set()
prev_vehicle_positions = {}

model = YOLO('yolov8n.pt')

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
                cv2.polylines(display_img, [np.array(points[-2:])], False, (0, 255, 255), 1)
            cv2.imshow("Define ROI", display_img)

    cv2.namedWindow("Define ROI")
    cv2.setMouseCallback("Define ROI", click_event)

    BOX_ROIS = {'traffic_light', 'detection_area'}

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

#VEHICLES DETECTION IN DETECTION AREA
def is_in_detection_area(point, da_pts):
    return cv2.pointPolygonTest(da_pts, (float(point[0]), float(point[1])), False) >= 0

def vehicles_detected(frame, da_pts, line_pts, status):
    global prev_vehicle_positions, violated_ids

    results = model.track(frame, conf=0.5, persist=True, tracker='bytetrack.yaml', verbose=False)
 
    if results is None or results[0].boxes is None:
        return frame, 0
 
    vehicle_count = 0
    boxes = results[0].boxes
 
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
 
        if cls_id not in VEHICLES:
            continue
 
        # FIX: akses track_id pakai index i langsung dari boxes.id
        track_id = None
        if boxes.id is not None:
            track_id = int(boxes.id[i])
 
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bottom = (x1, y2)
 
        if not is_in_detection_area(bottom, da_pts):
            continue
 
        vehicle_count += 1
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"
        if track_id is not None:
            label += f" id:{track_id}"

        is_violated = (track_id in violated_ids) if track_id is not None else False
        
        if (track_id is not None
                and status == 'red'
                and track_id not in violated_ids
                and track_id in prev_vehicle_positions):
 
            prev_pt = prev_vehicle_positions[track_id]
            delta_x = bottom[0] - prev_pt[0]

            is_moving_right = delta_x > 2

            if is_moving_right:
                if status == 'red' and is_crossing_line(prev_pt, bottom, line_pts):
                    violated_ids.add(track_id)
                    is_violated = True
                
        # Simpan posisi untuk frame berikutnya
        if track_id is not None:
            prev_vehicle_positions[track_id] = bottom

        box_color = (0, 0, 255) if is_violated else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.circle(frame, bottom, 4, box_color, -1)

    return frame, vehicle_count

#LINE DETECTION
def side_of_line(point, p1, p2):
    return ((p2[0]-p1[0]) * (point[1]-p1[1]) - (p2[1]-p1[1]) * (point[0]-p1[0]))

def is_crossing_line(prev_pt, curr_pt, line_pts):
    p1 = line_pts[0]
    p2 = line_pts[1]
    s_prev = side_of_line(prev_pt, p1, p2)
    s_curr = side_of_line(curr_pt, p1, p2)
 
    if s_prev == 0 or s_curr == 0:
        return True  # tepat di garis
 
    # Berpindah sisi?
    if (s_prev > 0) != (s_curr > 0):
        return True
 
    # Atau sangat dekat ke garis?
    x1, y1 = p1;  x2, y2 = p2
    px, py = curr_pt
    mag = np.hypot(x2-x1, y2-y1)
    if mag < 1e-6:
        return False
    dist = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / mag
    return dist < CROSS_DIST

#MAIN PART
def read_video_with_roi(roi):
    violation_count = 0
    line_pts = np.array(roi['line'], dtype=np.int32)
    
    cap = cv2.VideoCapture(PATH)
 
    if not cap.isOpened():
        raise ValueError(f"Gagal membuka video: {PATH}")
 
    da_pts = np.array(roi['detection_area'], dtype=np.int32)
    line_pts = np.array(roi['line'], dtype=np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame = cv2.resize(frame, (1280, 720))
        draw_roi_overlay(frame, roi)
 
        status = 'unknown'
        if 'traffic_light' in roi:
            status = detect_traffic_light_state(frame, roi['traffic_light'])
 
        frame, count = vehicles_detected(frame, da_pts, line_pts, status)
        cv2.putText(frame, f"Total Violations: {len(violated_ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Output", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    print(f"ID yang melanggar: {violated_ids}")

if __name__ == "__main__":
    first_frame = get_first_frame(PATH)
    roi = define_roi(first_frame)
    print(f"\nROI selesai: {list(roi.keys())}")

    read_video_with_roi(roi)
