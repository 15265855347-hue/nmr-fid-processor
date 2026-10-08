from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class NMRProcessingConfig:
    """Configuration for the NMR FID processing pipeline."""

    lb: float = 0.3
    ph0: float = 180.0
    ph1: float = 220.0
    reference_ppm: float = 1.25
    dwell_time_s: float = 1e-6
    spectral_width_ppm: float = 20.0
    baseline_window: int = 101
    baseline_polyorder: int = 3


def read_fid(path: str | Path) -> np.ndarray:
    """Load a 1D FID array from a .npy or .csv file."""

    file_path = Path(path)

    if file_path.suffix.lower() == ".npy":
        data = np.load(file_path)
    elif file_path.suffix.lower() in {".csv", ".txt"}:
        data = np.genfromtxt(file_path, delimiter=",", comments="#")
    else:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            "Use a .npy, .csv, or .txt file containing a 1D time-domain FID."
        )

    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 1:
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        else:
            raise ValueError("The FID data must be 1D after loading.")

    if array.size == 0:
        raise ValueError("Input FID is empty.")

    return array


def _apply_exponential_window(fid: np.ndarray, lb: float, dwell_time_s: float) -> np.ndarray:
    """Apply exponential apodization to the FID."""
    if lb <= 0:
        return fid.copy()

    n_points = fid.size
    time_axis = np.arange(n_points, dtype=np.float64) * dwell_time_s
    window = np.exp(-lb * np.pi * time_axis)
    return fid * window


def _fft_spectrum(fid: np.ndarray) -> np.ndarray:
    """Perform FFT and return the real spectrum in the frequency domain."""
    spectrum = np.fft.fft(fid)
    spectrum = np.fft.fftshift(spectrum)
    return spectrum


def _phase_correct(spectrum: np.ndarray, ph0: float, ph1: float) -> np.ndarray:
    """Apply zero-order and first-order phase correction."""
    n_points = spectrum.size
    index_axis = np.arange(n_points, dtype=np.float64) - n_points / 2.0
    normalized_axis = index_axis / max(n_points // 2, 1)
    phase = np.deg2rad(ph0 + ph1 * normalized_axis)
    phase_corrected = spectrum * np.exp(-1j * phase)
    return phase_corrected


def _baseline_correct(real_spectrum: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Estimate and subtract a smooth baseline from the spectrum."""
    if window <= 0 or real_spectrum.size < window:
        return real_spectrum.copy()

    odd_window = max(window // 2 * 2 + 1, 5)
    if odd_window > real_spectrum.size:
        odd_window = max(real_spectrum.size if real_spectrum.size % 2 == 1 else real_spectrum.size - 1, 5)

    baseline = savgol_filter(
        real_spectrum,
        window_length=odd_window,
        polyorder=min(polyorder, odd_window - 1),
        mode="interp",
    )
    return real_spectrum - baseline


def _ppm_axis(spectrum_size: int, spectral_width_ppm: float, reference_ppm: float) -> np.ndarray:
    """Build a ppm axis and shift the highest-intensity peak to the reference ppm."""
    ppm_axis = np.linspace(-spectral_width_ppm / 2.0, spectral_width_ppm / 2.0, spectrum_size)
    return ppm_axis


def _auto_reference(ppm_axis: np.ndarray, spectrum: np.ndarray, reference_ppm: float) -> np.ndarray:
    """Shift the ppm axis so that the largest peak aligns with the requested reference ppm."""
    peak_index = int(np.argmax(np.abs(spectrum)))
    current_peak_ppm = float(ppm_axis[peak_index])
    shift = current_peak_ppm - reference_ppm
    return ppm_axis - shift


def process_fid(
    fid: np.ndarray,
    lb: float = 0.3,
    ph0: float = 180.0,
    ph1: float = 220.0,
    reference_ppm: float = 1.25,
    dwell_time_s: float = 1e-6,
    spectral_width_ppm: float = 20.0,
    baseline_window: int = 101,
    baseline_polyorder: int = 3,
) -> dict[str, Any]:
    """Process a raw 1D FID into a phased, baseline-corrected, ppm-referenced spectrum."""

    input_fid = np.asarray(fid, dtype=np.float64)
    if input_fid.ndim != 1:
        raise ValueError("The input FID must be a 1D NumPy array.")

    apodized = _apply_exponential_window(input_fid, lb=lb, dwell_time_s=dwell_time_s)
    spectrum = _fft_spectrum(apodized)
    phased = _phase_correct(spectrum, ph0=ph0, ph1=ph1)

    real_part = np.real(phased)
    corrected = _baseline_correct(real_part, window=baseline_window, polyorder=baseline_polyorder)
    ppm_axis = _ppm_axis(corrected.size, spectral_width_ppm, reference_ppm)
    corrected_ppm = _auto_reference(ppm_axis, corrected, reference_ppm)

    return {
        "fid": input_fid,
        "apodized_fid": apodized,
        "spectrum": corrected,
        "ppm": corrected_ppm,
        "reference_ppm": reference_ppm,
        "config": {
            "lb": lb,
            "ph0": ph0,
            "ph1": ph1,
            "dwell_time_s": dwell_time_s,
            "spectral_width_ppm": spectral_width_ppm,
            "baseline_window": baseline_window,
            "baseline_polyorder": baseline_polyorder,
        },
    }


def export_csv(output_path: str | Path, ppm: np.ndarray, spectrum: np.ndarray) -> None:
    """Write the processed spectrum to a CSV file with ppm and intensity columns."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ppm", "spectrum"])
        for p, val in zip(ppm, spectrum, strict=False):
            writer.writerow([float(p), float(val)])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Process a 1D NMR FID file.")
    parser.add_argument("--input", required=True, help="Path to a .npy, .csv, or .txt FID file.")
    parser.add_argument("--output", default="processed_spectrum.csv", help="Path to the output CSV file.")
    parser.add_argument("--lb", type=float, default=0.3, help="Line broadening factor (default: 0.3).")
    parser.add_argument("--ph0", type=float, default=180.0, help="Zero-order phase correction in degrees.")
    parser.add_argument("--ph1", type=float, default=220.0, help="First-order phase correction in degrees.")
    parser.add_argument("--reference-ppm", type=float, default=1.25, help="Desired ppm of the reference peak.")
    parser.add_argument("--dwell-time-s", type=float, default=1e-6, help="Dwell time in seconds.")
    parser.add_argument("--spectral-width-ppm", type=float, default=20.0, help="Spectral width in ppm.")
    parser.add_argument("--baseline-window", type=int, default=101, help="Savitzky-Golay baseline window length.")
    parser.add_argument("--baseline-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
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
    print(f"Reference peak assigned to {args.reference_ppm} ppm.")


if __name__ == "__main__":
    main()
