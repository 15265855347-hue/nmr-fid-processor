"""NMR FID processing package."""

from .core import (
    NMRProcessingConfig,
    process_fid,
    read_fid,
)

__all__ = [
    "NMRProcessingConfig",
    "process_fid",
    "read_fid",
]
