<<<<<<< HEAD
import cv2
import os

output_folder = r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\images"
os.makedirs(output_folder, exist_ok=True)

videos = [
    r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Siang.mp4"
]

TARGET_FPS = 5 #berapa frame per detik yang ingin disimpan

total_saved = 0

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nMemproses: {video_name}")

    cap = cv2.VideoCapture(video_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval     = max(1, int(video_fps / TARGET_FPS))

    print(f"  FPS asli : {video_fps:.1f}")
    print(f"  Estimasi : ~{total_frames // interval} frame tersimpan")

    # Cari frame terakhir yang sudah tersimpan
    existing = [
        f for f in os.listdir(output_folder)
        if f.startswith(video_name) and f.endswith(".png")
    ]
    if existing:
        last_saved = max(int(f.split("_frame_")[1].replace(".png", "")) for f in existing)
        print(f"  Resume dari frame ke-{last_saved + 1}")
    else:
        last_saved = -1
        print("  Mulai dari awal")

    frame_idx = 0  # index frame di video (semua frame)
    saved     = 0  # index frame yang disimpan

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:        # ← sampling di sini
            if saved > last_saved:           # skip yang sudah ada (resume)
                filename = os.path.join(output_folder, f"{video_name}_frame_{saved:04d}.png")
                cv2.imwrite(filename, frame)
                if saved % 100 == 0:
                    print(f"  {saved} frame tersimpan...")
            saved += 1

        frame_idx += 1

    cap.release()
    added = saved - (last_saved + 1)
    print(f"  Selesai: +{added} frame baru dari '{video_name}'")
    total_saved += added

=======
<<<<<<< HEAD
import cv2
import os

output_folder = r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\images"
os.makedirs(output_folder, exist_ok=True)

videos = [
    r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Siang.mp4"
]

TARGET_FPS = 5 #berapa frame per detik yang ingin disimpan

total_saved = 0

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nMemproses: {video_name}")

    cap = cv2.VideoCapture(video_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval     = max(1, int(video_fps / TARGET_FPS))

    print(f"  FPS asli : {video_fps:.1f}")
    print(f"  Estimasi : ~{total_frames // interval} frame tersimpan")

    # Cari frame terakhir yang sudah tersimpan
    existing = [
        f for f in os.listdir(output_folder)
        if f.startswith(video_name) and f.endswith(".png")
    ]
    if existing:
        last_saved = max(int(f.split("_frame_")[1].replace(".png", "")) for f in existing)
        print(f"  Resume dari frame ke-{last_saved + 1}")
    else:
        last_saved = -1
        print("  Mulai dari awal")

    frame_idx = 0  # index frame di video (semua frame)
    saved     = 0  # index frame yang disimpan

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:        # ← sampling di sini
            if saved > last_saved:           # skip yang sudah ada (resume)
                filename = os.path.join(output_folder, f"{video_name}_frame_{saved:04d}.png")
                cv2.imwrite(filename, frame)
                if saved % 100 == 0:
                    print(f"  {saved} frame tersimpan...")
            saved += 1

        frame_idx += 1

    cap.release()
    added = saved - (last_saved + 1)
    print(f"  Selesai: +{added} frame baru dari '{video_name}'")
    total_saved += added

=======
import cv2
import os

output_folder = r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\images"
os.makedirs(output_folder, exist_ok=True)

videos = [
    r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Siang.mp4"
]

TARGET_FPS = 5 #berapa frame per detik yang ingin disimpan

total_saved = 0

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nMemproses: {video_name}")

    cap = cv2.VideoCapture(video_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval     = max(1, int(video_fps / TARGET_FPS))

    print(f"  FPS asli : {video_fps:.1f}")
    print(f"  Estimasi : ~{total_frames // interval} frame tersimpan")

    # Cari frame terakhir yang sudah tersimpan
    existing = [
        f for f in os.listdir(output_folder)
        if f.startswith(video_name) and f.endswith(".png")
    ]
    if existing:
        last_saved = max(int(f.split("_frame_")[1].replace(".png", "")) for f in existing)
        print(f"  Resume dari frame ke-{last_saved + 1}")
    else:
        last_saved = -1
        print("  Mulai dari awal")

    frame_idx = 0  # index frame di video (semua frame)
    saved     = 0  # index frame yang disimpan

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:        # ← sampling di sini
            if saved > last_saved:           # skip yang sudah ada (resume)
                filename = os.path.join(output_folder, f"{video_name}_frame_{saved:04d}.png")
                cv2.imwrite(filename, frame)
                if saved % 100 == 0:
                    print(f"  {saved} frame tersimpan...")
            saved += 1

        frame_idx += 1

    cap.release()
    added = saved - (last_saved + 1)
    print(f"  Selesai: +{added} frame baru dari '{video_name}'")
    total_saved += added

>>>>>>> 1fe3db47739574530fc5542d37e7e441f9b79124
>>>>>>> d74937f7a7a1e4841f8e70cfa4badba5141d4a38
print(f"\nTotal frame baru disimpan: {total_saved}")