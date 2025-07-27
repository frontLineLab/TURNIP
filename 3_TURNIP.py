import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error

def temporal_loss(y_true, y_pred):
    # reduce_sum: 内積
    # y_true, y_pred: shape (batch, T, 1)
    x = tf.squeeze(y_true, axis=-1)  # shape: (batch, T)
    z = tf.squeeze(y_pred, axis=-1)  # shape: (batch, T)
    T = tf.cast(tf.shape(x)[1], tf.float32)  # 時間長

    mu_x = tf.reduce_mean(x, axis=1, keepdims=True)  # shape: (batch, 1)
    mu_z = tf.reduce_mean(z, axis=1, keepdims=True)

    x_dot_z = tf.reduce_sum(x * z, axis=1, keepdims=True)  # shape: (batch, 1)
    x_dot_x = tf.reduce_sum(x * x, axis=1, keepdims=True)
    z_dot_z = tf.reduce_sum(z * z, axis=1, keepdims=True)

    numerator = T * x_dot_z - mu_x * mu_z
    denominator = tf.sqrt((T * x_dot_x - mu_x**2) * (T * z_dot_z - mu_z**2) + 1e-8)
    corr = numerator / denominator

    return 1.0 - tf.reduce_mean(corr)


# === TURNIP モデル定義 ===
def build_turnip_unet(input_length=300, channels=48):
    inp = layers.Input(shape=(input_length, channels))  # shape: [T, 48]

    def conv_gru_block(x, filters, kernel_size, strides=1):
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        x_conv = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(x)
        
        x_gru = layers.GRU(filters, return_sequences=True)(x)
        x_gru = layers.GRU(filters, return_sequences=True)(x_gru)
        return x, layers.Concatenate()([x_conv, x_gru])

    # --- Encoder ---
    # Level 1
    x0, e0 = conv_gru_block(inp, 64, kernel_size=3, strides=1)

    # Level 2
    x1, e1 = conv_gru_block(inp, 64, kernel_size=9, strides=3)

    # Level 3
    x2, e2 = conv_gru_block(x1, 128, kernel_size=7, strides=1)

    # Level 4
    x3, e3 = conv_gru_block(x2, 256, kernel_size=7, strides=2)

    # Level 5 (bottleneck)
    x4 = layers.Conv1D(512, kernel_size=7, strides=1, padding='same', activation='relu')(e3)
    x4 = layers.Conv1D(512, kernel_size=1, padding='same', activation='relu')(x4)
    x4 = layers.Dropout(0.3)(x4)
    x4 = layers.Conv1D(512, kernel_size=1, padding='same', activation='relu')(x4)

    # --- Decoder ---
    def upsample_concat_gru(x, skip, filters, upsample_size):
        x = layers.UpSampling1D(size=upsample_size)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv1D(filters, kernel_size=7, padding='same', activation='relu')(x)
        x = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(x)
        return x

    # d3 = upsample_concat_gru(x4, e3, 256)
    # d2 = upsample_concat_gru(d3, e2, 128)
    # Level 5 → Level 4 (e3, 50): そのままでOK (x4も50なので size=1)
    d3 = upsample_concat_gru(x4, e3, 256, upsample_size=1)

    # d3 (50) → e2 (100) に合わせる: 50→100 ⇒ size=2
    d2 = upsample_concat_gru(d3, e2, 128, upsample_size=2)

    # d2 (100) → e1 (300) に合わせる: 100→300 ⇒ size=3
    d1 = upsample_concat_gru(d2, e1, 64, upsample_size=1)
    # d1 = layers.UpSampling1D(size=3)(d2)
    # d1 = layers.Concatenate()([d1, e1])
    # d1 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(d1)
    # d1 = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu')(d1)
    
    d1 = layers.UpSampling1D(size=3)(d1)
    out = layers.Concatenate()([d1, x0])
    out = layers.Conv1D(64, kernel_size=7, strides=1, padding='same', activation='relu')(out)
    out = layers.Conv1D(1, kernel_size=1, strides=1, padding='same')(out)

    return tf.keras.models.Model(inputs=inp, outputs=out)


# === パス設定 ===
input_dir = "./output/output_windows"
gt_ppg_dir = "../../MR-NIRP/detect_facemesh_poly_driving_still_10s_0.5s/pulseOxy"
result_dir = "./output/turnip_results"
os.makedirs(result_dir, exist_ok=True)

fs = 30
win_size = 300

# === TURNIP モデル構築
model = build_turnip_unet(input_length=300, channels=48)
model.compile(optimizer='adam', loss=temporal_loss)
# model.load_weights("path_to_weights.h5")  # 重みがある場合はこちらを使用

# === 推論と評価
records = []
for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".csv"):
        continue

    input_path = os.path.join(input_dir, fname)
    x = pd.read_csv(input_path).values.astype(np.float32)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    x = np.expand_dims(x, axis=0)  # shape: (1, 300, 48)

    # 推定PPG
    pred_ppg = model.predict(x, verbose=0)[0, :, 0]  # shape: (300,)
    pred_ppg_path = os.path.join(result_dir, fname.replace(".csv", "_pred_ppg.csv"))
    pd.DataFrame(pred_ppg, columns=["pred_ppg"]).to_csv(pred_ppg_path, index=False)

    # HR 推定 (FFT)
    freqs, psd = welch(pred_ppg, fs=fs, nperseg=win_size)
    pred_bpm = freqs[np.argmax(psd)] * 60

    # === 正解PPG 読み込み
    gt_path = os.path.join(gt_ppg_dir, fname)
    gt_path = gt_path[:400]
    if os.path.exists(gt_path):
        gt_ppg = pd.read_csv(gt_path).values.flatten()
        if len(gt_ppg) != win_size:
            print(f"⚠ Skipping {fname} due to incorrect length.")
            continue
        # HR from GT
        freqs_gt, psd_gt = welch(gt_ppg, fs=fs, nperseg=win_size)
        gt_bpm = freqs_gt[np.argmax(psd_gt)] * 60

        # MAE 計算
        mae = mean_absolute_error(gt_ppg, pred_ppg)
    else:
        gt_ppg = None
        gt_bpm = np.nan
        mae = np.nan

    records.append({
        "window": fname,
        "predicted_hr_bpm": round(pred_bpm, 2),
        "ground_truth_hr_bpm": round(gt_bpm, 2),
        "mae_waveform": round(mae, 4)
    })

# === 結果保存
df = pd.DataFrame(records)
df.to_csv(os.path.join(result_dir, "ppg_estimation_results.csv"), index=False)
print("✅ 推定完了: ppg_estimation_results.csv を保存しました")
