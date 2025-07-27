import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from glob import glob
from scipy.signal import butter, filtfilt

# === パラメータ ===
input_dir = "/driving_still_940/sub1_driving_still_940/NIR"
output_csv = "./output/ppg_signals_48roi_filtered.csv"
fs = 30  # サンプリング周波数
lowcut = 0.75
highcut = 4.0

# === バンドパスフィルタ関数 ===
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# === ROIインデックス ===
roi_groups = [  # 48個
    (21, 54, 68, 71), (54, 103, 104, 68), (103, 67, 69, 104), (67, 109, 108, 69), (109, 10, 151, 108),
    (71, 68, 63, 70), (68, 104, 105, 63), (104, 69, 66, 105), (69, 108, 107, 66), (108, 151, 9, 107),
    (10, 338, 337, 151), (338, 297, 299, 337), (297, 332, 333, 299), (332, 284, 298, 333), (284, 251, 301, 298),
    (151, 337, 336, 9), (337, 299, 296, 336), (299, 333, 334, 296), (333, 298, 293, 334), (298, 301, 300, 293),
    (117, 101, 205, 50), (50, 205, 216, 187), (187, 216, 212, 214), (214, 212, 43, 210),
    (227, 117, 50, 137), (137, 50, 187, 177), (177, 187, 214, 58), (58, 214, 210, 172),
    (432, 434, 430, 273), (411, 436, 432, 434), (425, 280, 411, 436), (425, 330, 346, 280),
    (365, 430, 434, 367), (367, 434, 411, 401), (401, 411, 280, 352), (352, 280, 346, 345),
    (210, 43, 91, 204), (204, 91, 181, 194), (194, 181, 17, 200), (200, 17, 405, 418),
    (418, 405, 321, 424), (424, 321, 273, 430), (172, 210, 204, 170), (170, 204, 194, 140),
    (140, 194, 200, 152), (152, 200, 418, 400), (400, 418, 424, 379), (379, 424, 430, 365)
]

# === Mediapipe初期化 ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# === ファイル取得 ===
image_paths = sorted([f for f in glob(os.path.join(input_dir, "*.pgm")) if not os.path.basename(f).startswith("._")])

# === 平均輝度を抽出 ===
ppg_signals = []

for path in tqdm(image_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape

    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        ppg_signals.append([np.nan]*len(roi_groups))
        continue

    landmarks = results.multi_face_landmarks[0]
    lm_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

    roi_means = []
    for roi in roi_groups:
        pts = np.array([lm_points[i] for i in roi], np.int32)
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        roi_pixels = img[mask == 255]
        mean_intensity = float(np.mean(roi_pixels)) if roi_pixels.size > 0 else np.nan
        roi_means.append(mean_intensity)

    ppg_signals.append(roi_means)

# === DataFrame化 ===
df = pd.DataFrame(ppg_signals, columns=[f"ROI_{i}" for i in range(48)])

# === 前処理：線形補間 + バンドパス + 標準化
for col in df.columns:
    signal = df[col].interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    try:
        filtered = bandpass_filter(signal.values, fs, lowcut, highcut)
        standardized = (filtered - np.mean(filtered)) / np.std(filtered)
        df[col] = standardized
    except Exception as e:
        print(f"Error filtering {col}: {e}")
        df[col] = np.nan  # フィルタできなかった場合

# === 保存 ===
df.to_csv(output_csv, index=False)
print(f"Saved filtered and normalized PPG CSV: {output_csv}")
