import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def plot_12leads_final_pqrst_ptbxl(t, sig_f, results, leads, rec_id, fs, sec0=0.0, sec1=10.0, path=None):
    """
    動態全導程分析圖：支援不同導程數量與取樣頻率，具備高效能過濾機制。
    """
    n_leads = len(leads)
    fig, axes = plt.subplots(n_leads, 1, figsize=(15, 2 * n_leads), sharex=True)
    
    if n_leads == 1: axes = [axes]
    
    mask = (t >= sec0) & (t <= sec1)
    N_total = sig_f.shape[0] 
    
    for i, lname in enumerate(leads):
        ax = axes[i]
        lsig = sig_f[:, i]
        ax.plot(t[mask], lsig[mask], color='black', lw=0.6, alpha=0.3)
        
        res = results[lname]
        
        def safe_scatter(indices, color, marker, s, lw=1.0, zorder=5, is_gt=False):
            idx_arr = np.atleast_1d(indices).astype(int)
            valid = (idx_arr >= 0) & (idx_arr < N_total)
            v_idx = idx_arr[valid]
    
            if len(v_idx) > 0:
                time_s = v_idx / fs
                # 高效過濾：只計算視窗內附近的點
                in_view = (time_s >= sec0 ) & (time_s <= sec1 )
                v_idx = v_idx[in_view]
                time_s = time_s[in_view]
        
                if len(v_idx) > 0:
                    ax.scatter(time_s, lsig[v_idx], color=color, marker=marker, s=s, zorder=zorder)

        # 1. P 波峰與邊界 (紫色)
        safe_scatter(res['p_peaks'], 'purple', '^', 15, zorder=6)
        safe_scatter(res['p_ons'], 'purple', '|', 40, lw=0.8)
        safe_scatter(res['p_offs'], 'purple', '|', 40, lw=0.8)
        
        # 2. R 峰預測 (咖啡色實心點)
        safe_scatter(res['r_peaks'], 'brown', '.', s=20, zorder=7)
        
        # 3. QRS 邊界 (藍/紅槓)
        safe_scatter(res['q_ons'], 'blue', '|', 20, lw=0.05)
        safe_scatter(res['s_offs'], 'red', '|', 20, lw=0.05)
        
        # 4. T 波峰與邊界 (橘色)
        safe_scatter(res['t_peaks'], 'orange', '*', 20, zorder=6)
        safe_scatter(res['t_ons'], 'orange', '|', 45, lw=1)
        safe_scatter(res['t_offs'], 'orange', '|', 45, lw=1)
        

        ax.set_ylabel(f"{lname.upper()}", rotation=0, labelpad=25, fontsize=8)
        ax.grid(True, alpha=0.1)

    plt.suptitle(f"Record {rec_id}: Full Morphology Analysis ({fs}Hz)\n(Solid: Pred)", fontsize=16)
    
    # 存檔邏輯優化
    if path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"rec_{rec_id}_Full_PQRST_Morphology.png", dpi=300, bbox_inches='tight')
        print(f"✅ Full morphology plot saved to: {path}")
    
    # --- [關鍵新增] 釋放記憶體 ---
    plt.close()