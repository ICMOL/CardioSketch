# CardioSketch

CardioSketch is an ECG waveform analysis tool
for automatic P, QRS, and T wave detection.

## Supported Input

- XML ECG files
- WFDB records (.dat + .hea)

## Output

CardioSketch generates ECG waveform plots and measurement tables for each record.

ECG Plots : Visualizations of the detected PQRST waveforms across the 12 ECG leads.

Measurement : An Excel workbook containing 12 worksheets, one for each ECG lead. Each worksheet includes the detected waveform feature time points and amplitudes.

Measurement Summary : A single-sheet Excel workbook combining measurements from all 12 ECG leads. Each feature column is prefixed with its lead name (e.g., I_p_ons_time and II_p_ons_time), making it easier to analyze measurements across leads in one table.

## Installation

pip install -r requirements.txt

## Requirements

- Python 3.10.11 (tested)
- Dependencies listed in `requirements.txt`

## Usage

Run:

python main.py

The CardioSketch interface supports:

- Single Record: analyze one ECG record
- Batch Analysis: analyze multiple ECG records
