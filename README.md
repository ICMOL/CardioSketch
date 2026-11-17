# CardioSketch

CardioSketch is an ECG waveform analysis tool for automatic P, QRS, and T wave detection.

## Supported Input

- XML ECG files
- WFDB records (`.dat` + `.hea`)

## Output

CardioSketch generates ECG waveform plots and measurement tables for each record.

- **ECG Plots:** Visualizations of the detected PQRST waveforms across the 12 ECG leads.
- **Measurement:** An Excel workbook containing 12 worksheets, one for each ECG lead. Each worksheet includes the detected waveform feature time points and amplitudes.
- **Measurement Summary:** A single-sheet Excel workbook combining measurements from all 12 ECG leads. Each feature column is prefixed with its lead name (e.g., `I_p_ons_time` and `II_p_ons_time`), making it easier to analyze measurements across leads in one table.

## Download (Windows)

For users who want to run CardioSketch without installing Python:

1. Go to [GitHub Releases](https://github.com/ICMOL/CardioSketch/releases).
2. Download `CardioSketch.zip` from the latest release.
3. Extract the ZIP file.
4. Run `CardioSketch.exe`.

No Python installation is required.

**Note:** Windows SmartScreen may display a security warning because the application is not code-signed. If you downloaded it from the official GitHub Releases page, you can review the warning and choose to run the application.

## Run from Source

For users who want to run or modify the source code:

### Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10.11 (tested)
- Dependencies listed in `requirements.txt`

### Usage

Run:

```bash
python main.py
```

The CardioSketch interface supports:

- **Single Record:** Analyze one ECG record.
- **Batch Analysis:** Analyze multiple ECG records.
