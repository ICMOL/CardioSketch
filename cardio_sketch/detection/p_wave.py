import numpy as np
from scipy.signal import find_peaks
from cardio_sketch.utils.prominence import get_weighted_prominence
from cardio_sketch.utils.slope_scanner import gradient_scanner_v5
def detect_p_waves_with_lead_vote(lead_signal, q_ons, r_peaks, fs, l_name,
                                 last_t_offs=None, 
                                 all_leads_p_matrix=None, 
                                 wp_threshold=0.009, 
                                 p_outlier_threshold_ms=71): 
    """
    P 波偵測模組 (證據完整記錄 + 空視窗防護版)
    修正：解決 np.argmax 處理空陣列導致的 ValueError
    """ 
    n_beats = len(q_ons)
    p_gate_samples = (p_outlier_threshold_ms / 1000.0) * fs
    
    # --- [第一階段：極性判定] ---
    temp_results = [] 
    pos_wins, neg_wins = 0, 0
    for i in range(n_beats):

        curr_q_on = int(q_ons[i])

        # 原本的搜尋起點
        default_st = max(0, curr_q_on - int(0.375 * fs))

        st = default_st

        if (
            last_t_offs is not None
            and i > 0
            and last_t_offs[i - 1] > 0
        ):
            st = max(st, int(last_t_offs[i - 1]))

        en = curr_q_on

        segment = lead_signal[st:en]

        if len(segment) < 10:
            temp_results.append({
                'p': (0,0,0,0,0,0,0),
                'n': (0,0,0,0,0,0,0),
                'st': st,
                'en': en
            })
            continue
            
        pks_p, _ = find_peaks(segment); p_max = max([(p, *get_weighted_prominence(segment, p, fs=fs,base_search_ms=70)) for p in pks_p], key=lambda x: x[1], default=(0,0,0,0,0,0,0))
        pks_n, _ = find_peaks(-segment); n_max = max([(n, *get_weighted_prominence(-segment, n, fs=fs,base_search_ms=70)) for n in pks_n], key=lambda x: x[1], default=(0,0,0,0,0,0,0))
        if p_max[1] >= n_max[1]: pos_wins += 1
        else: neg_wins += 1
        temp_results.append({'p': p_max, 'n': n_max, 'st': st, 'en': en})

    global_lead_pol = 1 if pos_wins >= neg_wins else -1

    # --- [準備防呆 1：建立個人節律基準] ---
    all_rp_samples = []
    for i in range(n_beats):
        ph1 = temp_results[i]
        best_cand = ph1['p'] if global_lead_pol == 1 else ph1['n']
        if best_cand[1] > 0: 
            all_rp_samples.append(r_peaks[i] - (best_cand[0] + ph1['st']))
    typical_rp_samples = np.median(all_rp_samples) if all_rp_samples else 0

    p_peaks, p_proms, enriched_logs = [], [], []

    # --- [第二階段：精抓與 PK] ---
    for i in range(n_beats):
        curr_q_on = int(q_ons[i]); curr_r = int(r_peaks[i])
        ph1 = temp_results[i]
        
        # [防護 1]：確保精抓視窗合法
        st_p3 = ph1['st']
        en_p3 = max(st_p3, curr_q_on - int(0.0 * fs))
        
        target_sig = lead_signal[st_p3:en_p3] if global_lead_pol == 1 else -lead_signal[st_p3:en_p3]
        
        # 初始化變數
        winner = None
        pk_loser_abs = 0; pk_loser_wp = 0
        method = "Peak"; is_outlier = False; outlier_reason = ""; shift_ms = 0.0
        final_p_abs = 0; local_p_abs = 0
        
        pks, _ = find_peaks(target_sig)
        refined_list = []
        for p in pks:
            res = get_weighted_prominence(target_sig, p, fs=fs, ratio_threshold=0.06, use_scanner=False, ignore_trend=True,scan_params={'max_ms': 100, 'prot_ms': 10, 'n_sign_ms':16, 'n_trend_ms': 16},base_search_ms=70)
            refined_list.append({
                'idx': p, 'idx_abs': p + st_p3, 'wp': res[0], 'rat': res[5], 'pol': global_lead_pol
            })
        
        success_cands = sorted([c for c in refined_list if c['wp'] >= wp_threshold and c['rat'] >= 0.30], key=lambda x: x['wp'], reverse=True)

        if len(success_cands) > 0:
            winner = success_cands[0]
            local_p_abs = int(winner['idx_abs'])
            if len(success_cands) > 1:
                pk_loser_abs = int(success_cands[1]['idx_abs'])
                pk_loser_wp = success_cands[1]['wp']
            
            # 防呆 1：Temporal
            this_rp = curr_r - local_p_abs
            t_shift_samples = abs(this_rp - typical_rp_samples)
            is_temporal_outlier = t_shift_samples > p_gate_samples
            
            # 防呆 2：Spatial
            is_spatial_outlier = False; s_shift_samples = 0.0
            global_p_anchor = np.nan
            if all_leads_p_matrix is not None:
                global_p_anchor = np.nanmedian(all_leads_p_matrix[:, i])
                if not np.isnan(global_p_anchor):
                    s_shift_samples = abs(local_p_abs - global_p_anchor)
                    is_spatial_outlier = s_shift_samples > p_gate_samples

            if is_temporal_outlier or is_spatial_outlier:
                is_outlier = True; method = "Rescue_Outlier"
                outlier_reason = "T_Shift" if is_temporal_outlier else "S_Shift"
                shift_ms = (t_shift_samples if is_temporal_outlier else s_shift_samples) * 1000 / fs
                final_p_abs = int(global_p_anchor) if (all_leads_p_matrix is not None and not np.isnan(global_p_anchor)) else int(curr_r - typical_rp_samples)
            else:
                final_p_abs = local_p_abs; method = "Peak"; shift_ms = 0.0
        else:
            # [防護 2]：進入 Fallback 時，先檢查是否有集體共識，最後才用 argmax
            method = "Fallback"
            if all_leads_p_matrix is not None:
                valid_ps = all_leads_p_matrix[:, i][~np.isnan(all_leads_p_matrix[:, i])]
                if len(valid_ps) > 0:
                    final_p_abs = int(np.median(valid_ps))
                elif len(target_sig) > 0:
                    final_p_abs = int(np.argmax(target_sig) + st_p3)
                else:
                    final_p_abs = st_p3 # 極端空視窗防護
            else:
                if len(target_sig) > 0:
                    final_p_abs = int(np.argmax(target_sig) + st_p3)
                else:
                    final_p_abs = st_p3
            shift_ms = 0.0; local_p_abs = 0

        # 寫入結果
        final_p_abs = int(np.clip(final_p_abs, 0, len(lead_signal)-1))
        p_peaks.append(final_p_abs)
        
        # 修正：確保 winner 不是 None 才讀取 ['wp']
        curr_wp = winner['wp'] if (winner is not None and method == "Peak") else 0.0
        p_proms.append(curr_wp)
        
        enriched_logs.append({
            'lead': l_name, 'beat': i, 'method': method, 'is_outlier': is_outlier, 
            'reason': outlier_reason, 'shift_ms': round(shift_ms, 1),
            'final_p_abs': final_p_abs, 'local_p_abs': local_p_abs,
            'pk_loser_abs': pk_loser_abs, 'pk_loser_wp': pk_loser_wp,
            'wp': round(curr_wp, 5), 'st_en': (st_p3, en_p3),
            'ph1_p_abs': int(ph1['p'][0] + ph1['st']) if ph1['p'][1] > 0 else 0,
            'ph1_p_wp': ph1['p'][1],
            'ph1_n_abs': int(ph1['n'][0] + ph1['st']) if ph1['n'][1] > 0 else 0,
            'ph1_n_wp': ph1['n'][1]
        })

    return np.array(p_peaks), global_lead_pol, p_proms, enriched_logs
def detect_p_boundaries_all_leads(sig_f, p_results_pre, qrs_results, fs, leads, max_on_ms=100, prot_ms=35, overlap_ms=20,trend_ratio_on=0.3,trend_ratio_off=0.3,n_sign_ms=6,n_trend_ms=6,init_trend_ms=6):
    """
    P 波邊界偵測：利用 QRS Onset 作為 P-off 的動態搜尋邊界。
    overlap_ms: 允許 P-off 衝過 Q-on 的最大毫秒數 (設為 0 則絕不重疊)。
    """
    p_boundary_results = {}
    p_evidence_logs = []

    for l_idx, l_name in enumerate(leads):
        lead_sig = sig_f[:, l_idx]
        p_peaks = p_results_pre[l_name]['p_peaks']
        p_pol = p_results_pre[l_name]['p_polarity']
        # 拿到該導程對應的所有 Q-on
        q_ons = qrs_results[l_name]['q_ons']
        
        p_ons, p_offs = [], []
        
        for i, pp in enumerate(p_peaks):
            # 1. P-on 搜尋 (維持原狀)
            p_on_idx, p_on_re, p_on_win, p_on_it = gradient_scanner_v5(
                lead_sig, int(pp), -1, p_pol, fs, max_ms=max_on_ms, prot_ms=prot_ms,trend_ratio=trend_ratio_on, n_sign_ms=n_sign_ms, n_trend_ms=n_trend_ms, init_trend_ms=init_trend_ms
            )
            
            # 2. [關鍵修改] 動態計算 P-off 搜尋窗
            # 計算 P 峰到 Q-on 的距離，並換算成 ms
            dist_to_q = (q_ons[i] - pp) * 1000 / fs
            
            # 動態搜尋窗 = 距離 + 允許的重疊量 (若 dist_to_q 為負代表抓錯拍，保底給 50ms)
            dynamic_off_ms = max(50, dist_to_q + overlap_ms)
            
            p_off_idx, p_off_re, p_off_win, p_off_it = gradient_scanner_v5(
                lead_sig, int(pp), 1, p_pol, fs, max_ms=dynamic_off_ms, prot_ms=prot_ms,trend_ratio=trend_ratio_off, n_sign_ms=n_sign_ms, n_trend_ms=n_trend_ms, init_trend_ms=init_trend_ms
            )
            
            # 強制保險：如果 scanner 還是衝太遠，直接截斷在 Q-on + overlap 處
            limit_idx = q_ons[i] + int(overlap_ms * fs / 1000)
            if p_off_idx > limit_idx:
                p_off_idx = limit_idx
                p_off_re = "Forced_Limit_by_Qon"
            
            p_ons.append(p_on_idx)
            p_offs.append(p_off_idx)
            
            # 證據 Log (增加一個動態窗口的紀錄，方便 debug)
            p_evidence_logs.append({
                'lead': l_name, 
                'beat_id': i, 
                'p_peak': int(pp),
                'p_on': int(p_on_idx), 
                'p_off': int(p_off_idx),
                'p_on_reason': p_on_re,
                'p_off_reason': p_off_re,
                'q_on_ref': int(q_ons[i]), 
                'dynamic_win_ms': round(dynamic_off_ms, 1),
            })
            
        p_boundary_results[l_name] = {
            'p_ons': np.array(p_ons),
            'p_offs': np.array(p_offs)
        }
        
    return p_boundary_results, p_evidence_logs