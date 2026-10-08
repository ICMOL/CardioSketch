# CardioSketch

CardioSketch is an ECG waveform analysis tool
for automatic P, QRS, and T wave detection.

## Supported Input

- XML ECG files
- WFDB records (.dat + .hea)

## Output

The measurement table contains the detected time points
and amplitudes of ECG waveform features.


## Installation

pip install -r requirements.txt

## Usage

Run:

python main.py

The CardioSketch interface supports:

- Single Record: analyze one ECG record
- Batch Analysis: analyze multiple ECG records
