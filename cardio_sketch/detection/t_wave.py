import numpy as np
from scipy.signal import find_peaks
from cardio_sketch.utils.prominence import get_weighted_prominence
from cardio_sketch.utils.slope_scanner import gradient_scanner_v5
def detect_t_waves_with_lead_vote(lead_signal, s_offs, r_peaks, fs, l_name, 
                                 consensus_anchors=None, force_pol=None,
                                 normal_wp_threshold=0.013, rescue_wp_threshold=0.0165,
                                 outlier_threshold_ms=80, 
                                 show_debug_num=0, save_path=None, target_lead='v3', rescue_mode='Normal'):
    
    n_beats = len(s_offs)
    
    # --- 第一階段：極性公投 ---
    temp_candidates = []
    for i in range(n_beats):
        st_p1 = int(s_offs[i] + 0.00 * fs); en_p1 = min(len(lead_signal), st_p1 + int(0.550 * fs))
        if i < n_beats - 1:

            rr = r_peaks[i + 1] - r_peaks[i]

            rr60 = r_peaks[i] + int(rr * 0.55)

            en_p1 = min(en_p1, rr60)

            en_p1 = min(en_p1, int(r_peaks[i + 1] - 0.040 * fs))
        en_p1 = max(st_p1, en_p1)
        seg_p1 = lead_signal[st_p1:en_p1]
        if len(seg_p1) < 10: 
            temp_candidates.append({'p_wp':0, 'n_wp':0}); continue
        p_pks, _ = find_peaks(seg_p1); p_max = max([get_weighted_prominence(seg_p1, p, fs=fs,use_scanner=True,scan_params={'max_ms': 100, 'prot_ms': 15, 'n_sign_ms':20, 'n_trend_ms': 20},min_base_distance_ms=60)[0] for p in p_pks], default=0)
        n_pks, _ = find_peaks(-seg_p1); n_max = max([get_weighted_prominence(-seg_p1, n, fs=fs,use_scanner=True,scan_params={'max_ms': 100, 'prot_ms': 15, 'n_sign_ms':20, 'n_trend_ms': 20},min_base_distance_ms=60)[0] for n in n_pks], default=0)
        temp_candidates.append({'p_wp': p_max, 'n_wp': n_max})
    
    decided_t_pol = force_pol if force_pol is not None else (-1 if sum(1 for c in temp_candidates if c['n_wp'] > c['p_wp']) >= (n_beats / 2) else 1)

    t_peaks, t_proms, vote_logs = [], [], []
    debug_count = 0 

    for i in range(n_beats):
        rejected_loc_cand = None; opp_loc_peak = None 
        
        st = int(s_offs[i] + 0.00 * fs); en = st + int(0.550 * fs)
        st = max(0, st); en = min(en, len(lead_signal))
        if i < n_beats - 1:

            rr = r_peaks[i + 1] - r_peaks[i]

            rr60 = r_peaks[i] + int(rr * 0.55)

            en = min(en, rr60)

            en = min(en, int(r_peaks[i + 1] - 0.040 * fs))

        en = max(st, en) # 確保不為空
        
        local_win = lead_signal[st:en] 
        target_sig = local_win if decided_t_pol == 1 else -local_win
        
        # --- 第一輪：大窗海選 ---
        pks, _ = find_peaks(target_sig)
        best_cand = None
        if len(pks) > 0:
            all_p1_cands = []
            for p in pks:
                wp, hl, hr, lp, rp, rat = get_weighted_prominence(target_sig, p, fs=fs,use_scanner=True,scan_params={'max_ms': 100, 'prot_ms': 15, 'n_sign_ms':16, 'n_trend_ms': 16},min_base_distance_ms=80)
                all_p1_cands.append({'idx_abs': p + st, 'idx': p, 'wp': wp, 'l_pos': lp, 'r_pos': rp, 'ratio': rat, 'pol': decided_t_pol})
            best_cand = max(all_p1_cands, key=lambda x: x['wp'])


        f_idx = None; method = ""; curr_beat_pol = decided_t_pol

        
        if best_cand is not None:
            method = "Normal"; f_idx = best_cand['idx_abs']; curr_beat_pol = decided_t_pol
        else:
            method = "Normal_Argmax"
            f_idx = (np.argmax(target_sig) + st) if len(target_sig) > 0 else st
            curr_beat_pol = decided_t_pol

        f_idx = int(np.clip(f_idx, 0, len(lead_signal)-1))
        wp_f, hl_f, hr_f, lp_f, rp_f, rat_f = get_weighted_prominence(lead_signal if curr_beat_pol == 1 else -lead_signal, f_idx, fs=fs, use_scanner=True,scan_params={'max_ms': 100, 'prot_ms': 15, 'n_sign_ms':20, 'n_trend_ms': 20})
        # winner_data 的 idx 必須確保是給 plot 用的相對座標
        winner_data = {'idx': f_idx - st, 'wp': wp_f, 'l_pos': lp_f, 'r_pos': rp_f, 'ratio': rat_f, 'pol': curr_beat_pol}

        t_peaks.append(f_idx); t_proms.append(winner_data['wp'])
        is_strong = (method == "Normal") and (winner_data['wp'] >= normal_wp_threshold)

        # if l_name.lower() == target_lead.lower() and debug_count < show_debug_num:
        #     diag_path = save_path.parent / f"{save_path.stem}_{l_name}_beat{i}_diag.png" if save_path else None
        #     plot_t_calibration_pk_debug(local_win, winner_data, best_cand, opp_loc_peak, fs, i, l_name, 
        #                                curr_beat_pol, st, f_idx, None, save_path=diag_path,
        #                                is_strong=is_strong, is_outlier=False,
        #                                rejected_loc_cand=rejected_loc_cand, method=method, rescue_mode=rescue_mode)
        #     debug_count += 1

        vote_logs.append({

            'lead':l_name,

            'beat':i,

            'method':method,

            'decided_pol':curr_beat_pol,

            'wp':winner_data['wp'],

            'ratio':winner_data['ratio'],

            'final_t_abs':f_idx,

            'st_abs':st,

            'en_abs':en,

            'orig_sea_idx_abs':
                best_cand['idx_abs']
                if best_cand else None
        })
            
    return np.array(t_peaks), decided_t_pol, t_proms, vote_logs
def detect_t_boundaries_all_leads(sig_f, t_results, fs, leads,max_on_ms=150, max_off_ms=200, prot_ms=50,trend_ratio_on=0.8, trend_ratio_off=0.7,n_sign_ms=6, n_trend_ms=6, init_trend_ms=6):
    """
    T 波邊界偵測：利用 T 波極性，從 T 峰雙向執行梯度搜尋。
    規則：Sign Flip (2點), Trend Break (3點)。
    """
    t_boundary_results = {}
    t_evidence_logs = []

    for l_idx, l_name in enumerate(leads):
        lead_sig = sig_f[:, l_idx]
        t_peaks = t_results[l_name]['t_peaks']
        t_pol = t_results[l_name]['t_polarity']
        
        t_ons, t_offs = [], []
        
        for i, tp in enumerate(t_peaks):
            # --- T 側 Onset 搜尋 (往左) ---
            # 從 T 峰往左尋找趨勢斷裂或翻轉
            t_on_idx, t_on_re, t_on_win, t_on_it = gradient_scanner_v5(lead_sig, int(tp), -1, t_pol, fs, max_ms=max_on_ms,prot_ms=prot_ms,trend_ratio=trend_ratio_on, n_sign_ms=n_sign_ms, n_trend_ms=n_trend_ms, init_trend_ms=init_trend_ms)
            
            # --- T 側 Offset 搜尋 (往右) ---
            # 從 T 峰往右尋找趨勢斷裂或翻轉
            t_off_idx, t_off_re, t_off_win, t_off_it = gradient_scanner_v5(lead_sig, int(tp), 1, t_pol, fs, max_ms=max_off_ms,prot_ms=prot_ms,trend_ratio=trend_ratio_off, n_sign_ms=n_sign_ms, n_trend_ms=n_trend_ms, init_trend_ms=init_trend_ms)
            
            t_ons.append(t_on_idx)
            t_offs.append(t_off_idx)
            
            # 建立 T 波專屬證據 Log，避免跟 QRS 混在一起爆炸
            t_evidence_logs.append({
                'lead': l_name,
                'beat_id': i,
                't_peak': int(tp),
                't_polarity': t_pol,
                't_on': int(t_on_idx),
                't_on_reason': t_on_re,
                't_on_slopes': str(t_on_win),
                't_off': int(t_off_idx),
                't_off_reason': t_off_re,
                't_off_slopes': str(t_off_win)
            })
            
        t_boundary_results[l_name] = {
            't_ons': np.array(t_ons),
            't_offs': np.array(t_offs)
        }
        
    return t_boundary_results, t_evidence_logs
def fallback_t_peak_consensus(
    t_pre,
    r_all,
    leads,
    fs,
    tol_ms=40,
    score_ratio=0.5,
    trusted_ratio=0.9
):
    import numpy as np
    
    tol = int(tol_ms * fs / 1000)

    # ----------------------------
    # 建 matrix
    # ----------------------------
    t_matrix = np.array([
        t_pre[l]["t_peaks"]
        for l in leads
    ], dtype=int)

    r_matrix = np.array([
        r_all[l]
        for l in leads
    ], dtype=int)

    rt_matrix = t_matrix - r_matrix

    n_leads, n_beats = t_matrix.shape

    row_score = np.zeros_like(t_matrix)
    col_score = np.zeros_like(t_matrix)

    # ----------------------------
    # Row score
    # ----------------------------
    for l in range(n_leads):

        for b in range(n_beats):

            rt0 = rt_matrix[l, b]

            row_score[l, b] = np.sum(
                np.abs(rt_matrix[l] - rt0) <= tol
            )

    # ----------------------------
    # Column score
    # ----------------------------
    for b in range(n_beats):

        for l in range(n_leads):

            t0 = t_matrix[l, b]

            col_score[l, b] = np.sum(
                np.abs(t_matrix[:, b] - t0) <= tol
            )

    total_score = row_score + col_score

    # ----------------------------
    # 自適應 threshold
    # ----------------------------
    max_score = np.max(total_score)

    rescue_threshold = max_score * score_ratio

    trusted_threshold = max_score * trusted_ratio

    rescue_cnt = 0

    # print("\n========== Fallback ==========")

    # ----------------------------
    # Rescue
    # ----------------------------
    for b in range(n_beats):

        beat_max_score = np.max(total_score[:, b])

        trusted_threshold = beat_max_score * trusted_ratio

        trusted_mask = total_score[:, b] >= trusted_threshold

        trusted_rt = rt_matrix[:, b][trusted_mask]

        if len(trusted_rt) < 2:
            continue

        consensus_rt = int(np.median(trusted_rt))

        for l in range(n_leads):

            if total_score[l, b] > rescue_threshold:
                continue

            old_t = t_matrix[l, b]

            t_matrix[l, b] = r_matrix[l, b] + consensus_rt

            # print(
            #     f"{leads[l]:4s}"
            #     f" Beat {b:3d}"
            #     f" Score={total_score[l,b]:2d}"
            #     f"  {old_t:5d} -> {t_matrix[l,b]:5d}"
            # )

            rescue_cnt += 1

    # print(f"\nRescued {rescue_cnt} points")

    # ----------------------------
    # 寫回 t_pre
    # ----------------------------
    for l_idx, l_name in enumerate(leads):

        t_pre[l_name]["t_peaks"] = t_matrix[l_idx]

    return t_pre