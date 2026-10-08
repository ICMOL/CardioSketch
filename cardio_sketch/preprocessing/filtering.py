import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
# =========================
# 濾波（零相位）
# =========================
def wavelet_denoising_transient(sig, fs, wavelet='db8', level=8,max_ratio_D23=0.6):
    """
    DWT 降噪：加入數據偵測點，用於診斷閾值設定問題。
    此版本強制零化 D1，並對 D2/D3 使用激進硬閾值。
    """
    
    # 確保輸入是二維陣列
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]

    sig_denoised = np.zeros_like(sig, dtype=float)
    num_leads = sig.shape[1]

    # # --- 關鍵診斷數據紀錄 ---
    # print("\n--- DWT 數據診斷報告 ---")
    
    for i in range(num_leads):
        x = sig[:, i]
        coeffs = pywt.wavedec(x, wavelet, level=level)
        new_coeffs = coeffs.copy()
        
        # 1. 處理 D1 (強制零化)
        if len(coeffs) > 1:
             new_coeffs[-1] = np.zeros_like(new_coeffs[-1])
        
        # 2. 處理 D2, D3 (使用層次最大值比例作為閾值)
        num_levels_to_treat = 2 # D2 和 D3
        start_j = len(coeffs) - num_levels_to_treat - 1 
        end_j = len(coeffs) - 1 

        for j in range(start_j, end_j):
            detail = coeffs[j]
            level_name = f"D{level - (j - 1)}" 
            
            # 【關鍵修改點：閾值設定】
            # 將閾值設定為該層次最大振幅的 max_ratio_D23 倍。
            # 閾值 T = Max(|D_j|) * max_ratio_D23
            threshold = np.max(np.abs(detail)) * max_ratio_D23 
            
            # 診斷輸出
            # print(f"導聯 {i+1} - {level_name}: Max振幅 {np.max(np.abs(detail)):.6e}, 閾值 T {threshold:.6e}")
            
            # 使用硬閾值
            new_coeffs[j] = pywt.threshold(detail, threshold, mode='hard') 
            
        # 3. IDWT 重建
        # --- 修改後的安全版本 ---
        reconstructed = pywt.waverec(new_coeffs, wavelet)
        # 強制裁切至原始訊號 x 的長度，解決重建長度偏差問題
        sig_denoised[:, i] = reconstructed[:len(x)]

    return sig_denoised
def zero_phase_bandpass(x, fs, lo=0.5, hi=40.0, order=3):
    """
    0.5–40 Hz 帶通（含去基線），零相位；x 可是 (N, L) 或 (N,)。
    """
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=0)  # 零相位


def zero_phase_notch(x, fs, f0=50.0, Q=30.0):
    """
    50/60 Hz 陷波（視需要），零相位。
    """
    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    return filtfilt(b, a, x, axis=0)  # 零相位