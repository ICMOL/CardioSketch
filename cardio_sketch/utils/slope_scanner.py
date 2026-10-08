import numpy as np
def gradient_scanner_v5(signal, start_idx, direction, polarity, fs, 
                        max_ms=100, prot_ms=10, n_sign_ms=6, n_trend_ms=6,trend_ratio=0.1,init_trend_ms=6):
    """
    參數化掃描引擎：
    - n_sign: Sign Flip 需要連續變號的點數
    - n_trend: Trend Break 需要連續平緩的點數
    """
    n_sign = max(1, round(n_sign_ms / 1000 * fs))
    n_trend = max(1, round(n_trend_ms / 1000 * fs))
    init_trend_window = max(1, round(init_trend_ms / 1000 * fs))
    slopes = np.diff(signal) 
    N_slopes = len(slopes)
    limit_samples = round((max_ms / 1000.0) * fs)
    prot_samples = round((prot_ms / 1000.0) * fs)
    
    # 初始趨勢設定
    if direction == -1:

        init_win = slopes[
            max(0, start_idx-init_trend_window):
            start_idx
        ]

    else:
        init_win = slopes[
            start_idx:
            min(
                N_slopes,
                start_idx+init_trend_window
            )
        ]
    
    init_trend = np.mean(init_win) * polarity if len(init_win) > 0 else 0
    if np.abs(init_trend) < 1e-5:
        return start_idx, "Trend Break (Initial)", [], init_trend

    curr_idx = start_idx
    for i in range(start_idx + direction, start_idx + direction * (limit_samples + 1), direction):
        # 【修改 1】：保護機制只看 n_sign，不要被 n_trend 撐爆
        if direction == -1:
            if i < n_sign or i >= N_slopes: break
            win_s = slopes[i - (n_sign-1) : i+1]
            win_t = slopes[i - (n_trend-1) : i+1] if i >= (n_trend-1) else [] # 防止越界
        else:
            if i < 0 or i > N_slopes - n_sign: break # 這裡拿掉 max
            win_s = slopes[i : i + n_sign]
            win_t = slopes[i : i + n_trend] if i <= N_slopes - n_trend else []

        curr_idx = i 
        if abs(i - start_idx) <= prot_samples: continue

        # 1. Sign Flip 判定 (維持原樣)
        if np.all(np.sign(win_s * polarity) != np.sign(init_trend)):
            return i, "Sign Flip", win_s, init_trend
            
        # 【修改 2】：增加判斷，只有 n_trend > 0 且 win_t 有東西時才執行
        if n_trend > 0 and len(win_t) == n_trend:
            if np.all(np.abs(win_t * polarity) < trend_ratio * np.abs(init_trend)):
                return i, "Trend Break", win_t, init_trend
            
    return curr_idx, "Limit", [], init_trend