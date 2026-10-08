from __future__ import annotations

import argparse
from pathlib import Path

from nmr_fid_processor.core import export_csv, process_fid, read_fid


def main() -> None:
    parser = argparse.ArgumentParser(description="Process 1D NMR FID data.")
    parser.add_argument("--input", required=True, help="Path to the raw FID file (.npy, .csv, or .txt).")
    parser.add_argument("--output", default="processed_spectrum.csv", help="Output CSV file path.")
    parser.add_argument("--lb", type=float, default=0.3, help="Exponential line broadening factor.")
    parser.add_argument("--ph0", type=float, default=180.0, help="Zero-order phase correction in degrees.")
    parser.add_argument("--ph1", type=float, default=220.0, help="First-order phase correction in degrees.")
    parser.add_argument("--reference-ppm", type=float, default=1.25, help="Reference ppm position.")
    parser.add_argument("--dwell-time-s", type=float, default=1e-6, help="Dwell time in seconds.")
    parser.add_argument("--spectral-width-ppm", type=float, default=20.0, help="Spectral width in ppm.")
    parser.add_argument("--baseline-window", type=int, default=101, help="Window length for baseline smoothing.")
    parser.add_argument("--baseline-polyorder", type=int, default=3, help="Polynomial order for baseline smoothing.")
    args = parser.parse_args()

    fid = read_fid(args.input)
    processed = process_fid(
        fid=fid,
        lb=args.lb,
        ph0=args.ph0,
        ph1=args.ph1,
        reference_ppm=args.reference_ppm,
        dwell_time_s=args.dwell_time_s,
        spectral_width_ppm=args.spectral_width_ppm,
        baseline_window=args.baseline_window,
        baseline_polyorder=args.baseline_polyorder,
    )

    export_csv(args.output, processed["ppm"], processed["spectrum"])
    print(f"Processed spectrum written to: {args.output}")


if __name__ == "__main__":
    main()
