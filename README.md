# NMR FID Processor

This repository implements a simple NMR FID processing pipeline inspired by the workflow you described:

- Raw FID
- Exponential windowing (LB = 0.3)
- Fourier transform
- Phase correction (ph0 = 180°, ph1 = 220°)
- Baseline correction
- Chemical shift reference at 1.25 ppm

The code is designed as a reusable Python library and a command-line tool.

## Features

- Read 1D FID data from `.npy` or `.csv` files
- Exponential apodization using a configurable line broadening factor
- FFT-based spectrum generation
- Frequency-domain phase correction with zero- and first-order terms
- Baseline correction via Savitzky-Golay smoothing
- Automatic ppm re-referencing to a target peak
- Output of processed spectrum and metadata

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### CLI usage

```bash
python -m nmr_fid_processor.cli \
  --input ./example_data/fid.npy \
  --output ./processed_spectrum.csv \
  --lb 0.3 \
  --ph0 180 \
  --ph1 220 \
  --reference-ppm 1.25 \
  --spectral-width-ppm 20 \
  --dwell-time-s 1e-6
```

### Python usage

```python
from nmr_fid_processor import process_fid

fid = ...  # 1D NumPy array
result = process_fid(
    fid=fid,
    lb=0.3,
    ph0=180.0,
    ph1=220.0,
    reference_ppm=1.25,
    dwell_time_s=1e-6,
    spectral_width_ppm=20.0,
)

print(result["ppm"][:10])
print(result["spectrum"][:10])
```

## Input data notes

- For a simple workflow, store the raw FID as a 1D NumPy array or CSV column.
- `dwell_time_s` is required for the exponential window function.
- If you are working with Bruker/Varian data, this repository can be extended with format-specific loaders such as `nmrglue`.

## Example output

The script writes a CSV file with columns:

```csv
ppm,spectrum
-10.0,0.0123
-9.95,0.0132
...
```

## License

MIT
