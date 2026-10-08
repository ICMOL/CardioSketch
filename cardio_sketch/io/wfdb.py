from pathlib import Path

import wfdb


def load_wfdb_record(input_path):
    """Load a WFDB record from either its .dat or .hea file.

    The .dat and .hea files must be in the same directory and share the
    same base name. WFDB itself reads the pair when given the record base
    path (without an extension).
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix not in {".dat", ".hea"}:
        raise ValueError("Please select a WFDB .dat or .hea file.")

    dat_path = input_path.with_suffix(".dat")
    hea_path = input_path.with_suffix(".hea")

    missing = []
    if not dat_path.exists():
        missing.append(dat_path.name)
    if not hea_path.exists():
        missing.append(hea_path.name)

    if missing:
        raise FileNotFoundError(
            "This WFDB record is incomplete. Missing: "
            + ", ".join(missing)
        )

    record = wfdb.rdrecord(str(input_path.with_suffix("")))

    return {
        "sig": record.p_signal,
        "fs": record.fs,
        "leads": record.sig_name,
    }
