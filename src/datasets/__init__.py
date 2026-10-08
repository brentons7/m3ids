"""--dataset name -> module with RAW_DIRNAME and prepare(raw_dir, out_dir)."""
from . import ciciomt2024

REGISTRY = {
    "ciciomt2024": ciciomt2024,
}
