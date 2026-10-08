from .slope_scanner import gradient_scanner_v5
import numpy as np
def get_weighted_prominence(signal, peak_idx, w_left=0.5, w_right=0.5, ratio_threshold=0.06, 
                            fs=500, use_scanner=False, scan_params=None,ignore_trend=False, min_base_distance_ms=None,base_search_ms=None):
    """
    計算 WP 分數。
    use_scanner: 是否改用趨勢掃描尋找波形邊界 (Base)。
    scan_params: 字典，包含 {'max_ms', 'prot_ms', 'n_sign', 'n_trend'}
    """
    if peak_idx <= 0 or peak_idx >= len(signal) - 1:
        return 0, 0, 0, 0, 0, 0
    
    if not use_scanner:
        # --- 傳統模式：找指定搜尋窗內絕對最低點 ---
        if base_search_ms is None:
            left_st = 0
            right_en = len(signal)
        else:
            half_win = int(base_search_ms * fs / 1000)
            left_st = max(0, peak_idx - half_win)
            right_en = min(len(signal), peak_idx + half_win + 1)

        left_seg = signal[left_st:peak_idx]
        if len(left_seg) > 0:
            v_l_idx = left_st + np.argmin(left_seg)
        else:
            v_l_idx = peak_idx

        right_seg = signal[peak_idx + 1:right_en]
        if len(right_seg) > 0:
            v_r_idx = (peak_idx + 1) + np.argmin(right_seg)
        else:
            v_r_idx = peak_idx
    else:
        # --- 趨勢模式：利用 Scanner 找邊界 ---
        p = scan_params if scan_params else {'max_ms': 100, 'prot_ms': 15, 'n_sign_ms':12, 'n_trend_ms': 6}
        if ignore_trend:
            p['n_trend_ms'] = 0
        # 往左掃 (direction=-1)，從峰頂往下走，預期斜率是負的 (若往左看)
        # 這裡 polarity 設 1 代表遵循初始趨勢
        v_l_idx, _, _, _ = gradient_scanner_v5(signal, peak_idx, -1, 1, fs, **p)
        
        # 往右掃 (direction=1)
        v_r_idx, _, _, _ = gradient_scanner_v5(signal, peak_idx, 1, 1, fs, **p)

    h_l = signal[peak_idx] - signal[v_l_idx]
    h_r = signal[peak_idx] - signal[v_r_idx]
    if min_base_distance_ms is not None:
        min_dist = int(min_base_distance_ms * fs / 1000)
        if (v_r_idx - v_l_idx) < min_dist:
            return 0, h_l, h_r, (v_l_idx - peak_idx), (v_r_idx - peak_idx), 0
    # 計算對稱比例
    min_ratio = min(h_l / h_r, h_r / h_l) if (h_l > 0 and h_r > 0) else 0

    if min_ratio < ratio_threshold:
        wp = 0
    else:
        wp = (w_left * h_l) + (w_right * h_r)
        
    return wp, h_l, h_r, (v_l_idx - peak_idx), (v_r_idx - peak_idx), min_ratio