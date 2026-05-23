import cv2
import os

output_folder = r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\images"
os.makedirs(output_folder, exist_ok=True)

videos = [
    r"D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Malam.mp4"
]

TARGET_FPS = 1            # 1 frame per detik (1 detik 1 foto)
SKIP_FIRST_FRAMES = 20    # Skip 20 frame di awal video

total_saved = 0

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nMemproses: {video_name}")

    cap = cv2.VideoCapture(video_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Interval otomatis dihitung berdasarkan FPS video agar pas 1 detik
    interval     = max(1, int(video_fps / TARGET_FPS))

    print(f"  FPS asli : {video_fps:.1f}")
    print(f"  Interval : Ambil 1 frame setiap {interval} frame")
    print(f"  Estimasi : ~{(total_frames - SKIP_FIRST_FRAMES) // interval} frame tersimpan")

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

        # 1. Skip 20 frame pertama
        if frame_idx < SKIP_FIRST_FRAMES:
            frame_idx += 1
            continue

        # Kurangi index dengan jumlah frame yang di-skip agar perhitungan 1 detik dimulai dari sini
        adjusted_idx = frame_idx - SKIP_FIRST_FRAMES

        # 2. Ambil 1 foto setiap interval (1 detik)
        if adjusted_idx % interval == 0:        
            if saved > last_saved:              # skip yang sudah ada (resume)
                filename = os.path.join(output_folder, f"{video_name}_frame_{saved:04d}.png")
                cv2.imwrite(filename, frame)
                
                # Menampilkan log setiap 10 frame tersimpan (diubah karena 1 detik 1 foto lebih lama)
                if saved % 10 == 0:
                    print(f"  {saved} frame tersimpan...")
            saved += 1

        frame_idx += 1

    cap.release()
    added = saved - (last_saved + 1)
    print(f"  Selesai: +{added} frame baru dari '{video_name}'")
    total_saved += added

print(f"\nTotal frame baru disimpan: {total_saved}")