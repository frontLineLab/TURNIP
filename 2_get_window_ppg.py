import os
import numpy as np
import pandas as pd

# === パラメータ ===
input_csv = "./output/ppg_signals_48roi_filtered.csv"
output_dir = "./output/output_windows"
os.makedirs(output_dir, exist_ok=True)

fs = 30  # サンプリング周波数 (Hz)
win_sec = 10
slide_sec = 0.5
win_size = int(fs * win_sec)       # 10秒 → 300フレーム
slide_size = int(fs * slide_sec)   # 0.5秒 → 15フレーム

# === データ読み込み ===
df = pd.read_csv(input_csv)
ppg_array = df.values  # shape: (T, 48)
T, D = ppg_array.shape

# === ウィンドウ分割・保存 ===
window_index = 0
for start in range(0, T - win_size + 1, slide_size):
    segment = ppg_array[start:start + win_size]
    if np.isnan(segment).any():
        continue  # 欠損があるウィンドウはスキップ

    # 保存
    out_path = os.path.join(output_dir, f"window_{window_index:04d}.csv")
    pd.DataFrame(segment, columns=[f"ROI_{i}" for i in range(D)]).to_csv(out_path, index=False)
    window_index += 1

print(f"Saved {window_index} windowed CSV files to: {output_dir}")
