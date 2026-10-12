import numpy as np
from pathlib import Path
from cardio_sketch.preprocessing.filtering import wavelet_denoising_transient, zero_phase_bandpass, zero_phase_notch
from cardio_sketch.detection.r_peak import detect_r_peaks_simple, refine_peaks_on_signal,detect_r_all_leads_from_ref
from cardio_sketch.detection.qrs import detect_qrs_all_leads_v3
from cardio_sketch.detection.t_wave import detect_t_waves_with_lead_vote, detect_t_boundaries_all_leads, fallback_t_peak_consensus
from cardio_sketch.detection.p_wave import detect_p_waves_with_lead_vote, detect_p_boundaries_all_leads
from cardio_sketch.measurement.measurement import build_measurement
from cardio_sketch.visualization.plot import plot_12leads_final_pqrst_ptbxl
from cardio_sketch.io.xml.chung_long import load_chung_long_xml
from cardio_sketch.io.wfdb import load_wfdb_record
def run_pipeline_for_record(rec_id):

    input_path = Path(rec_id)
    rec_id = input_path.stem
    save_pqrst = Path("output") / "ecg_plots"
    output_folder = Path("output") / "measurement"
    summary_folder = Path("output") / "measurement_summary"
    save_pqrst.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    summary_folder.mkdir(parents=True, exist_ok=True)
    try:
        # --- A. 載入數據 ---
        # XML and WFDB share the same downstream pipeline. Only the
        # source-specific loader is selected here; everything after this
        # point operates on the same (sig, fs, leads) representation.
        suffix = input_path.suffix.lower()

        if suffix == ".xml":
            sig, fs, leads = load_chung_long_xml(input_path)

        elif suffix in {".dat", ".hea"}:
            # A WFDB record is the .dat + .hea pair. The loader accepts
            # either side of the pair, so the user does not need to know
            # which one is the "correct" file to select.
            wfdb_data = load_wfdb_record(input_path)
            sig = wfdb_data["sig"]
            fs = wfdb_data["fs"]
            leads = wfdb_data["leads"]

        else:
            raise ValueError(
                f"Unsupported input format: {input_path.suffix or '(no extension)'}"
            )

        t = np.arange(sig.shape[0]) / fs
        

        # --- B. 濾波與 R 偵測 ---
        sig_f = zero_phase_notch(zero_phase_bandpass(wavelet_denoising_transient(sig, fs), fs, 0.5, 40), fs, 50)
        # ==========================================================
        # Select reference lead by RR-std consensus
        # ==========================================================

        lead_infos = []

        for l_idx, l_name in enumerate(leads):

            x_f = sig_f[:, l_idx]

            peaks_c, _ = detect_r_peaks_simple(x_f, fs)

            peaks_f, _ = refine_peaks_on_signal(
                x_f,
                peaks_c,
                fs,
                win_ms=60
            )

            if len(peaks_f) < 3:
                continue

            lead_infos.append({
                "lead": l_name,
                "peaks": peaks_f,
                "rr_std": np.std(np.diff(peaks_f)),
                "beat_count": len(peaks_f)
            })

        if len(lead_infos) == 0:
            raise RuntimeError("No valid reference lead found.")

        # ----------------------------------------------------------
        # Filter by beat count
        # ----------------------------------------------------------

        median_count = int(np.median([x["beat_count"] for x in lead_infos]))

        candidate_infos = [
            x for x in lead_infos
            if x["beat_count"] >= median_count - 1
        ]

        if len(candidate_infos) == 0:
            candidate_infos = lead_infos

        # ----------------------------------------------------------
        # RR std consensus grouping
        # ----------------------------------------------------------

        candidate_infos.sort(key=lambda x: x["rr_std"])

        RR_STD_RATIO = 0.05      # within 5% regarded as same group

        groups = []
        current_group = [candidate_infos[0]]

        for info in candidate_infos[1:]:

            if info["rr_std"] <= current_group[0]["rr_std"] * (1 + RR_STD_RATIO):
                current_group.append(info)
            else:
                groups.append(current_group)
                current_group = [info]

        groups.append(current_group)

        best_group = max(groups, key=len)
        # ----------------------------------------------------------
        # Select median member of the largest group
        # ----------------------------------------------------------

        best_info = best_group[len(best_group) // 2]

        best_peaks = best_info["peaks"]
        best_lead = best_info["lead"]
        r_all, r_pol, _ = detect_r_all_leads_from_ref(sig_f, fs, leads, r_ref=best_peaks)
        
        # --- C. QRS & P & T 偵測 (含投票) ---
        qrs_results, _ = detect_qrs_all_leads_v3(sig_f, r_all, r_pol, fs, leads)
        # --- D. T 波 & P 波 雙階段救援 (這才是靈魂) ---
        # ======================================================
        # Pass 1：先抓 T Peak
        # ======================================================
        t_pre = {}
        for l_idx, l_name in enumerate(leads):

            lsig = sig_f[:, l_idx]

            tp, tpol, _, _ = detect_t_waves_with_lead_vote(
                lsig,
                qrs_results[l_name]["s_offs"],
                r_all[l_name],
                fs,
                l_name
            )
            t_pre[l_name] = {
                "t_peaks": tp,
                "t_polarity": tpol
            }


        t_pre = fallback_t_peak_consensus(
            t_pre,
            r_all,
            leads,
            fs,
            tol_ms=40,
            score_ratio=0.5,
            trusted_ratio=0.9
        )
        for l_name in leads:

            qrs_results[l_name]["t_peaks"] = t_pre[l_name]["t_peaks"]
        # ======================================================
        # T Boundary
        # ======================================================
        t_bounds, _ = detect_t_boundaries_all_leads(
            sig_f,
            t_pre,
            fs,
            leads
        )

        for ln in leads:
            qrs_results[ln].update(t_bounds[ln])


        # ======================================================
        # Pass 2：利用上一拍 T off 抓 P
        # ======================================================
        for l_idx, l_name in enumerate(leads):

            lsig = sig_f[:, l_idx]

            pp, ppol, _, _ = detect_p_waves_with_lead_vote(
                lsig,
                qrs_results[l_name]["q_ons"],
                r_all[l_name],
                fs,
                l_name,
                last_t_offs=qrs_results[l_name]["t_offs"]
            )

            qrs_results[l_name].update({
                "p_peaks": pp,
                "p_polarity": ppol
            })


        # ======================================================
        # P Boundary
        # ======================================================
        p_bounds, _ = detect_p_boundaries_all_leads(
            sig_f,
            qrs_results,
            qrs_results,
            fs,
            leads
        )

        for ln in leads:
            qrs_results[ln].update(p_bounds[ln])
        plot_12leads_final_pqrst_ptbxl(t, sig_f, qrs_results, leads, rec_id, fs, sec0=0.0, sec1=10, path=save_pqrst)
        measurement = build_measurement(
            rec_id=rec_id,
            fs=fs,
            leads=leads,
            sig_f=sig_f,
            qrs_results=qrs_results,
            best_lead=best_lead
        )

        import pandas as pd
        import os

        output_path = output_folder / f"{rec_id}_measurement.xlsx"
        summary_path = summary_folder / f"{rec_id}_measurement_summary.xlsx"

        # Version 1: one worksheet per lead (keep the existing format).
        # Version 2: one worksheet with lead-prefixed feature columns.
        summary_rows_by_beat = {}

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for lead, lead_data in measurement["lead_results"].items():
                # Each lead may have a different number of detected beats.
                max_beats = max(
                    (len(point_data["idx"]) for point_data in lead_data.values()),
                    default=0,
                )

                rows = []
                for beat_idx in range(max_beats):
                    row = {
                        "Record ID": measurement["record_id"],
                        "Lead": lead,
                        "Beat": beat_idx + 1,
                    }

                    for point, point_data in lead_data.items():
                        indices = point_data["idx"]
                        amplitudes = point_data["amp"]

                        if beat_idx < len(indices):
                            idx = indices[beat_idx]
                            amp = amplitudes[beat_idx]
                            row[f"{point}_time"] = (
                                idx / measurement["fs"] * 1000
                            )
                            row[f"{point}_amp"] = amp
                        else:
                            row[f"{point}_time"] = None
                            row[f"{point}_amp"] = None

                    rows.append(row)

                    # Align all leads by Record ID + Beat.
                    # Prefix every measurement column with its lead name.
                    summary_key = (measurement["record_id"], beat_idx + 1)
                    summary_row = summary_rows_by_beat.setdefault(
                        summary_key,
                        {
                            "Record ID": measurement["record_id"],
                            "Beat": beat_idx + 1,
                        },
                    )
                    for column_name, value in row.items():
                        if column_name in {"Record ID", "Lead", "Beat"}:
                            continue
                        summary_row[f"{lead}_{column_name}"] = value

                pd.DataFrame(rows).to_excel(
                    writer,
                    sheet_name=lead[:31],
                    index=False,
                )

        # Save the summary workbook as a single worksheet.
        summary_df = pd.DataFrame(list(summary_rows_by_beat.values()))
        if not summary_df.empty:
            # Use the conventional 12-lead ECG order instead of alphabetical order.
            lead_order = [
                "I", "II", "III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6",
            ]
            lead_rank = {lead: index for index, lead in enumerate(lead_order)}

            def summary_column_sort_key(column_name):
                # Column names have the form "<Lead>_<feature>", e.g. "I_p_ons_time".
                for lead in lead_order:
                    prefix = f"{lead}_"
                    if column_name.startswith(prefix):
                        return (lead_rank[lead], column_name[len(prefix):])
                # Keep any unexpected columns after the standard 12 leads.
                return (len(lead_order), column_name)

            feature_columns = sorted(
                (
                    column for column in summary_df.columns
                    if column not in {"Record ID", "Beat"}
                ),
                key=summary_column_sort_key,
            )
            summary_df = summary_df[["Record ID", "Beat"] + feature_columns]

        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return measurement
        


    except Exception as e:
        # 1. 記得要在函式開頭或這裡 import traceback
        import traceback 
        
        print("\n" + "!"*60)
        print(f"❌ 偵測到崩潰！Record ID: {rec_id}")
        print(f"錯誤簡述: {str(e)}")
        print("-" * 60)
        
        # 2. 核心代碼：印出詳細的錯誤行數與原因
        traceback.print_exc() 
        
        print("!"*60 + "\n")
        return None
def run_pipeline_for_folder(folder_path, progress_callback=None):
    """
    批次處理資料夾中的 ECG。
    支援：
    - XML：每個 .xml 視為一筆 record
    - WFDB：同名 .dat + .hea 視為一筆 record

    每筆 record 都交給既有的 run_pipeline_for_record() 處理。
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Not a valid folder: {folder_path}"
        )

    records = []

    # ----------------------------------------------------------
    # 1. XML
    # ----------------------------------------------------------
    for xml_path in sorted(folder_path.glob("*.xml")):
        records.append(xml_path)

    # ----------------------------------------------------------
    # 2. WFDB
    #    .dat + .hea 同名才算一筆完整 record
    # ----------------------------------------------------------
    for dat_path in sorted(folder_path.glob("*.dat")):
        hea_path = dat_path.with_suffix(".hea")

        if hea_path.exists():
            records.append(dat_path)
        else:
            print(
                f"⚠️ Skip WFDB record '{dat_path.stem}': "
                f"missing {hea_path.name}"
            )

    if not records:
        raise FileNotFoundError(
            "No supported ECG records were found in this folder."
        )

    print("\n" + "=" * 60)
    print(f"Batch processing started")
    print(f"Folder: {folder_path}")
    print(f"Records found: {len(records)}")
    print("=" * 60)

    results = {}
    failed = []

    # ----------------------------------------------------------
    # 3. 逐筆執行既有 pipeline
    # ----------------------------------------------------------
    for i, record_path in enumerate(records, start=1):

        print("\n" + "-" * 60)
        print(f"[{i}/{len(records)}] Processing: {record_path.name}")
        print("-" * 60)

        result = run_pipeline_for_record(record_path)

        if result is not None:
            results[record_path.stem] = result
        else:
            failed.append(record_path.stem)

        # Report progress after each record finishes. The GUI callback should
        # only enqueue this information; Tkinter widgets are updated on the
        # main thread.
        if progress_callback is not None:
            progress_callback({
                "completed": i,
                "total": len(records),
                "current": record_path.name,
                "success": len(results),
                "failed": len(failed),
            })

    # ----------------------------------------------------------
    # 4. Batch summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch processing completed")
    print(f"Successful: {len(results)}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print("\nFailed records:")
        for rec_id in failed:
            print(f"  - {rec_id}")

    print("=" * 60 + "\n")

    return {
        "results": results,
        "failed": failed,
        "total": len(records),
    }