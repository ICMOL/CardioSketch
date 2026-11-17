
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    # Running as a packaged executable
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    # Running from source (python main.py)
    BASE_DIR = Path(__file__).resolve().parent.parent


OUTPUT_DIR = BASE_DIR / "output"

ECG_PLOTS_DIR = OUTPUT_DIR / "ecg_plots"
MEASUREMENT_DIR = OUTPUT_DIR / "measurement"
SUMMARY_DIR = OUTPUT_DIR / "measurement_summary"
