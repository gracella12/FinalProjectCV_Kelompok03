import cv2
import numpy as np
import os

from ultralytics import YOLO

PATH = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Malam.mp4'
model = YOLO('yolo11n.pt')
ROI_NAMES = {'traffic_light', 'detection_area','line'}

#Read File
def read_video():
    cap = cv2.VideoCapture(PATH)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize (frame, (1280, 720))
        cv2.imshow('Frame', frame) #read frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Gagal membaca frame pertama dari '{video_path}'")
    return cv2.resize(frame, (1280, 720))
    
def define_roi(frame):
    points      = []   # titik sementara ROI yang sedang diklik
    roi         = {}   # hasil akhir {nama: [list titik]}
    display_img = frame.copy()

    def click_event(event, x, y, flags, params):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Gambar titik dan garis ke display_img (bukan frame asli)
            cv2.circle(display_img, (x, y), 5, (0, 255, 255), -1)
            if len(points) >= 2:
                cv2.line(display_img, points[-2], points[-1], (0, 255, 255), 2)
            cv2.imshow("Define ROI", display_img)

    cv2.namedWindow("Define ROI")
    cv2.setMouseCallback("Define ROI", click_event)

    for name in ROI_NAMES:
        points.clear()

        # Reset display_img dari frame bersih + gambar ROI yang sudah tersimpan
        display_img = frame.copy()
        for saved_name, saved_pts in roi.items():
            pts   = np.array(saved_pts, dtype=np.int32)
            cv2.polylines(display_img, [pts], isClosed=True, color=(255,0,0), thickness=2)
            cv2.putText(display_img, saved_name, tuple(saved_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Instruksi di layar
        cv2.putText(display_img,
                    f"ROI: {name}  |  klik titik  |  's' = simpan & lanjut",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        print(f"\n[{name}] Klik titik-titik ROI, tekan 's' jika sudah selesai.")
        cv2.imshow("Define ROI", display_img)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('s') and len(points) >= 2:
                break

        # Tutup polygon 
        cv2.line(display_img, points[-1], points[0], (0, 255, 255), 2)
        cv2.imshow("Define ROI", display_img)
        cv2.waitKey(400)

        # Simpan ROI ini
        roi[name] = points.copy()

    cv2.destroyWindow("Define ROI")
    return roi

def read_video_with_roi(roi):
    cap = cv2.VideoCapture(PATH)

    #conveert to points
    tl_pts = np.array(roi['traffic_light'], dtype=np.int32)
    da_pts = np.array(roi['detection_area'], dtype=np.int32)
    line_pts = np.array(roi['line'], dtype=np.int32)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        cv2.polylines(frame, [tl_pts], isClosed=True, color=(0,255,255), thickness=2)
        cv2.polylines(frame, [da_pts], isClosed=True, color=(255,255,0), thickness=2)
        cv2.polylines(frame, [line_pts], isClosed=True, color=(255,0,255), thickness=2)

        cv2.imshow('Frame with ROI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def apply_mask(frame, tl_pts, da_pts):
    mask_tl = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_da = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_tl, [tl_pts], 255)
    cv2.fillPoly(mask_da, [da_pts], 255)
    masked_tl = cv2.bitwise_and(frame, frame, mask=mask_tl)
    masked_da = cv2.bitwise_and(frame, frame, mask=mask_da)
    return masked_tl, masked_da

def detection (frame):
    results = model(frame)
    return results

if __name__ == "__main__":
    first_frame = get_first_frame(PATH)
    roi_definitions = define_roi(first_frame)

    mased_tl, masked_da = apply_mask(first_frame,
                                     np.array(roi_definitions['traffic_light'], dtype=np.int32),
                                     np.array(roi_definitions['detection_area'], dtype=np.int32))
    
