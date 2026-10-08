from cardio_sketch.utils.prominence import gradient_scanner_v5
def detect_qrs_all_leads_v3(sig_f, r_all, r_polarity_logs, fs, leads):
    """
    更新版主控：QRS 邊界偵測 (aVR 特殊過濾版)。
    [邏輯更新]：
    1. 振幅過濾規則 (0.175/0.275) 現在被「侷限在 aVR 導程」。
    2. 其他導程 (如 I, II, V1~V6) 只要抓到 Sign Flip 或 Trend Break 就會直接採用。
    3. aVR 依然維持 while 迴圈，過濾掉微小的雜訊起伏直到找到真正的波峰。
    """
    import numpy as np
    results_dict = {}
    detailed_logs = []
    
    pol_lookup = {e['lead_name']: (1 if e['decided_polarity'] == 'Positive' else -1) for e in r_polarity_logs}
    
    for l_idx, l_name in enumerate(leads):
        lead_sig = sig_f[:, l_idx]
        r_peaks = r_all[l_name]
        p = pol_lookup.get(l_name, 1)
        
        # 統一轉成大寫方便比對
        is_avr = (l_name.upper() == 'AVR')
        
        q_peaks, s_peaks, q_ons, s_offs = [], [], [], []
        
        for i, r in enumerate(r_peaks):
            # --- Q 側搜尋 (向左搜尋 -1) ---
            curr_q_start = r
            while True:
                q_evt, q_re1, q_win1, q_it1 = gradient_scanner_v5(lead_sig, curr_q_start, -1, p, fs, 70, 10,10, 18,trend_ratio=0.2)
                
                if q_re1 in ["Sign Flip", "Trend Break"]:
                    # 【關鍵修改點】：只有 aVR 才檢查 0.175 門檻
                    if is_avr and abs(lead_sig[r] - lead_sig[q_evt]) < 0.175:
                        curr_q_start = q_evt - 1 # 忽略此微小點，繼續往左搜尋
                        if curr_q_start < 0:
                            q_peak_idx, q_on_idx, q_re2, q_win2, q_it2 = -1, q_evt, "N/A", [], 0.0
                            break
                        continue # 重新執行第一階段搜尋
                    
                    # 若不是 aVR，或振幅已達標
                    if q_re1 == "Sign Flip":
                        q_peak_idx = q_evt
                        q_on_idx, q_re2, q_win2, q_it2 = gradient_scanner_v5(lead_sig, q_peak_idx, -1, p, fs, 30, 0,10, 18,trend_ratio=0.2)
                    else: # Trend Break
                        q_peak_idx, q_on_idx, q_re2, q_win2, q_it2 = -1, q_evt, "N/A", [], 0.0
                    break
                else:
                    q_peak_idx, q_on_idx, q_re2, q_win2, q_it2 = -1, q_evt, "N/A", [], 0.0
                    break

            # --- S 側搜尋 (向右搜尋 1) ---
            curr_s_start = r
            while True:
                s_evt, s_re1, s_win1, s_it1 = gradient_scanner_v5(lead_sig, curr_s_start, 1, p, fs, 70, 30, 10, 18,trend_ratio=0.4)
                
                if s_re1 in ["Sign Flip", "Trend Break"]:
                    # 【關鍵修改點】：只有 aVR 才檢查 0.275 門檻
                    if is_avr and abs(lead_sig[r] - lead_sig[s_evt]) < 0.275:
                        curr_s_start = s_evt + 1 # 忽略此微小點，繼續往右搜尋
                        if curr_s_start >= len(lead_sig):
                            s_peak_idx, s_off_idx, s_re2, s_win2, s_it2 = -1, s_evt, "N/A", [], 0.0
                            break
                        continue # 重新執行第一階段搜尋
                    
                    # 若不是 aVR，或振幅已達標
                    if s_re1 == "Sign Flip":
                        s_peak_idx = s_evt
                        s_off_idx, s_re2, s_win2, s_it2 = gradient_scanner_v5(lead_sig, s_peak_idx, 1, p, fs, 60, 0,trend_ratio=0.2)
                    else: # Trend Break
                        s_peak_idx, s_off_idx, s_re2, s_win2, s_it2 = -1, s_evt, "N/A", [], 0.0
                    break
                else:
                    s_peak_idx, s_off_idx, s_re2, s_win2, s_it2 = -1, s_evt, "N/A", [], 0.0
                    break
            
            # --- 存入結果 ---
            q_peaks.append(q_peak_idx); q_ons.append(q_on_idx)
            s_peaks.append(s_peak_idx); s_offs.append(s_off_idx)
            
            detailed_logs.append({
                'lead': l_name, 'beat_id': i,
                'q1_re': q_re1, 'q1_len': len(q_win1),
                's1_re': s_re1, 's1_len': len(s_win1)
            })
            
        results_dict[l_name] = {
            'r_peaks': np.array(r_peaks),
            'q_peaks': np.array(q_peaks), 
            's_peaks': np.array(s_peaks),
            'q_ons': np.array(q_ons), 
            's_offs': np.array(s_offs)
        }
        
    return results_dict, detailed_logs