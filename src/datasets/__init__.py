"""--dataset name -> module with RAW_DIRNAME and prepare(raw_dir, out_dir, val_frac)."""
from . import ciciomt2024, iomtind2026, iomttrafficdata, xiomt

REGISTRY = {
    "ciciomt2024": ciciomt2024,
    "iomtind2026": iomtind2026,
    "xiomt": xiomt,
    "iomttrafficdata": iomttrafficdata,
}
