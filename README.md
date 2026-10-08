# CardioSketch

CardioSketch is an ECG waveform analysis tool
for automatic P, QRS, and T wave detection.

## Supported Input

- XML ECG files
- WFDB records (.dat + .hea)

## Output

CardioSketch generates:

- P-wave peak and boundaries
- QRS boundaries and R peaks
- T-wave peak and boundaries
- Timing and amplitude measurements
- PQRST visualization

## Installation

pip install -r requirements.txt

## Usage

python main.py