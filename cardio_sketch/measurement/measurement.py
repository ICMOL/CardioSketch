import numpy as np
def build_measurement(rec_id, fs, leads, sig_f, qrs_results, best_lead=None):
    """
    將 delineation 結果整理成 Measurement Database。

    每個 fiducial point 儲存：
        - idx : sample index
        - amp : 該 sample 的濾波後電位
    """

    measurement = {
        "record_id": rec_id,
        "fs": fs,
        "reference_lead": best_lead,
        "lead_results": {},
        "signal_length": sig_f.shape[0],
    }

    point_keys = [
        "p_ons",
        "p_peaks",
        "p_offs",
        "q_ons",
        "r_peaks",
        "s_offs",
        "t_ons",
        "t_peaks",
        "t_offs"
    ]

    for l_idx, lead in enumerate(leads):

        sig = sig_f[:, l_idx]
        res = qrs_results[lead]

        lead_dict = {}

        for key in point_keys:

            idx = np.asarray(res.get(key, []), dtype=int)

            # 避免超出訊號範圍
            idx = idx[(idx >= 0) & (idx < len(sig))]

            lead_dict[key] = {
                "idx": idx.tolist(),
                "amp": sig[idx].tolist()
            }

        measurement["lead_results"][lead] = lead_dict

    return measurement