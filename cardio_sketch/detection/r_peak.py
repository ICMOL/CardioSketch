from cardio_sketch.preprocessing.filtering import zero_phase_bandpass
from cardio_sketch.utils.prominence import get_weighted_prominence
import numpy as np
from scipy.signal import find_peaks
def detect_r_peaks_simple(x, fs):
    """
    粗抓階段：回傳 R 點索引、全域診斷資訊與每一拍的包絡強度。
    """
    # 1) 5–18 Hz 帶通與平方
    x_bp = zero_phase_bandpass(x, fs, lo=5.0, hi=18.0, order=3)
    y = x_bp ** 2

    # 2) 150ms 滑動平均包絡
    win = int(0.15 * fs)
    kernel = np.ones(win) / win
    env = np.convolve(y, kernel, mode='same')

    # 3) 設定門檻
    # --- 新寫法：抵抗雜訊的穩健門檻 ---
    # 我們取所有點中「前 10% 強度」的位準作為基準 (90th Percentile)
    ref_val = np.percentile(env, 90) 
    # 以這個位準的一半作為門檻，這樣就算後面有超大雜訊，門檻也不會跑掉
    height_threshold = 0.30 * ref_val
    distance = int(0.40 * fs)

    # 4) 找峰值
    peaks_env, properties = find_peaks(env, distance=distance, height=height_threshold)
    
    # --- 建立粗抓日誌 ---
    coarse_logs = {
        'global_ref_val': ref_val,  # 整段訊號包絡的 90th Percentile 強度
        'threshold': height_threshold,
        'peak_heights': properties['peak_heights'], # 每一拍抓到的包絡高度
        'env_signal': env  # 回傳整段包絡供後續繪圖檢查
    }
    
    return peaks_env, coarse_logs
   
def refine_peaks_on_signal(x, peaks_env, fs, win_ms=40):
    """
    R 點精修：採用兩段式極性鎖定邏輯，並補全繪圖所需的日誌數據。
    """
    from scipy.signal import find_peaks
    import numpy as np
    
    w = int(win_ms * fs / 1000)
    N = len(x)
    
    # -------------------------------------------------------
    # 第一階段：極性公投 (Referendum)
    # -------------------------------------------------------
    pos_wins = 0
    neg_wins = 0
    temp_beat_candidates = []

    for p in peaks_env:
        i0 = max(0, p - w)
        i1 = min(N, p + w + 1)
        window_data = x[i0:i1]
        
        p_cands, _ = find_peaks(window_data)
        n_cands, _ = find_peaks(-window_data)
        
        # 找出正負最強代表
        p_best = p_cands[np.argmax(window_data[p_cands])] if len(p_cands) > 0 else None
        n_best = n_cands[np.argmin(window_data[n_cands])] if len(n_cands) > 0 else None
        
        # 統計極性傾向
        if p_best is not None and n_best is not None:
            if abs(window_data[p_best]) >= abs(window_data[n_best]): pos_wins += 1
            else: neg_wins += 1
        elif p_best is not None: pos_wins += 1
        elif n_best is not None: neg_wins += 1
        
        temp_beat_candidates.append({'p_idx': p_best, 'n_idx': n_best, 'i0': i0, 'i1': i1, 'win': window_data})

    # 決定全導程極性
    global_pol = "pos" if pos_wins >= neg_wins else "neg"
    # print(f">>> Global Polarity Lock: {global_pol.upper()}")

    # -------------------------------------------------------
    # 第二階段：強制極性精抓與日誌紀錄
    # -------------------------------------------------------
    refined_indices = []
    beat_logs = []

    for i, data in enumerate(temp_beat_candidates):
        i0 = data['i0']
        window_data = data['win']
        p_idx = data['p_idx']
        n_idx = data['n_idx']
        
        best_local_idx = None
        is_slope_inverted = False # 用來記錄是否抓到波峰（而非用極端值備援）
        
        if global_pol == "pos":
            if p_idx is not None:
                best_local_idx = p_idx
                is_slope_inverted = True
            else:
                best_local_idx = np.argmax(window_data)
                is_slope_inverted = False
        else:
            if n_idx is not None:
                best_local_idx = n_idx
                is_slope_inverted = True
            else:
                best_local_idx = np.argmin(window_data)
                is_slope_inverted = False

        final_idx = int(best_local_idx + i0)
        coarse_p = peaks_env[i]
        
        # --- 補齊繪圖所需的 Key ---
        beat_logs.append({
            'beat_id': i,
            'coarse_idx': coarse_p,
            'refined_idx': final_idx,
            'offset_samples': final_idx - coarse_p,      # 修正 KeyError 的關鍵
            'detected_polarity': global_pol,
            'local_amplitude': window_data[best_local_idx],
            'slope_inversion': is_slope_inverted,        # 繪圖時區分顏色用
            'search_window': (i0, data['i1'])            # 繪圖時畫視窗用
        })
        refined_indices.append(final_idx)

    return np.array(refined_indices, dtype=int), beat_logs
def detect_r_all_leads_from_ref(sig_f, fs, leads, r_ref, win_ms=120):
    """
    R 波多導程同步校準 (WP 全細節追蹤版)
    修正點：解決第二階段變數 seg 殘留導致的索引偏移 Bug。
    """
    from scipy.signal import find_peaks
    import numpy as np
    
    r_ref = np.asarray(r_ref, dtype=int)
    N, L = sig_f.shape
    w = int(win_ms * fs / 1000)
    w = max(1, w)

    r_all, polarity_logs, r_beat_logs = {}, [], []

    for li, lname in enumerate(leads):
        xk = sig_f[:, li]
        temp_phase1 = [] 
        pos_wins, neg_wins = 0, 0
        total_net_wp = 0.0
        
        # --- 第一階段：極性公投與候選人收集 ---
        for r in r_ref:
            i0, i1 = max(0, r - w), min(N, r + w + 1)
            seg = xk[i0:i1]
            if len(seg) < 3:
                temp_phase1.append({'p': [0]*6, 'n': [0]*6, 'st': i0, 'en': i1})
                continue
            
            # 正向最強
            p_pks, _ = find_peaks(seg)
            p_cands = [(p, *get_weighted_prominence(seg, p)) for p in p_pks]
            best_p = max(p_cands, key=lambda x: x[1]) if p_cands else (0, 0, 0, 0, 0, 0)
            
            # 負向最強
            n_pks, _ = find_peaks(-seg)
            n_cands = [(n, *get_weighted_prominence(-seg, n)) for n in n_pks]
            best_n = max(n_cands, key=lambda x: x[1]) if n_cands else (0, 0, 0, 0, 0, 0)
            
            if best_p[1] >= best_n[1]: pos_wins += 1
            else: neg_wins += 1
            total_net_wp += (best_p[1] - best_n[1])
            # 多存一個 en 確保第二階段視窗一致
            temp_phase1.append({'p': best_p, 'n': best_n, 'st': i0, 'en': i1})
        
        decided_p_pol = -1 if neg_wins > pos_wins else 1
        polarity_logs.append({
            'lead_name': lname, 'pos_wins': pos_wins, 'neg_wins': neg_wins,
            'total_net_wp': round(total_net_wp, 5),
            'decided_polarity': 'Positive' if decided_p_pol == 1 else 'Negative'
        })

        # --- 第二階段：根據公投結果執行最終精抓 ---
        for beat_idx, r in enumerate(r_ref):
            ph1 = temp_phase1[beat_idx]
            st, en = ph1['st'], ph1['en']
            
            # 修正處：重新獲取 target_sig，不再依賴外部迴圈殘留的 seg
            current_seg = xk[st:en]
            target_sig = current_seg if decided_p_pol == 1 else -current_seg
            
            # 最終精抓候選人
            final_pks, _ = find_peaks(target_sig)
            final_cands = sorted([
                (p, *get_weighted_prominence(target_sig, p)) for p in final_pks
            ], key=lambda x: x[1], reverse=True)
            
            beat_info = {
                'lead': lname, 'beat_idx': beat_idx, 'ref_r': int(r),
                'lead_decided_pol': 'pos' if decided_p_pol == 1 else 'neg',
                # Phase 1 正向細節 (用於畫圖)
                'ph1_pos_abs': ph1['p'][0] + st if ph1['p'][0] > 0 else 0,
                'ph1_pos_wp': round(ph1['p'][1], 5),
                'ph1_pos_L_abs': ph1['p'][0] + ph1['p'][4] + st,
                'ph1_pos_R_abs': ph1['p'][0] + ph1['p'][5] + st,
                # Phase 1 負向細節 (用於畫圖)
                'ph1_neg_abs': ph1['n'][0] + st if ph1['n'][0] > 0 else 0,
                'ph1_neg_wp': round(ph1['n'][1], 5),
                'ph1_neg_L_abs': ph1['n'][0] + ph1['n'][4] + st,
                'ph1_neg_R_abs': ph1['n'][0] + ph1['n'][5] + st,
                # 最終輸出結果
                'final_r': 0, 'shift_ms': 0,"window_start": int(st),
                "window_end": int(en),
            }

            if final_cands:
                f_idx = final_cands[0][0] + st
                beat_info.update({'final_r': int(f_idx), 'shift_ms': round((f_idx - r) * 1000 / fs, 2)})
                r_val = int(f_idx)
            else:
                # 保底：若沒抓到 peak 則用 argmax
                r_val = st + np.argmax(target_sig) if len(target_sig) > 0 else st
                beat_info.update({'final_r': int(r_val), 'shift_ms': round((r_val - r) * 1000 / fs, 2)})
            
            r_beat_logs.append(beat_info)
            if lname not in r_all: r_all[lname] = []
            r_all[lname].append(r_val)
            
    for ln in leads: r_all[ln] = np.array(r_all[ln], dtype=int)
    return r_all, polarity_logs, r_beat_logs