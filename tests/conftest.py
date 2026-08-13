from __future__ import annotations

import sys
from pathlib import Path

# Keep the tests runnable from either the combined TypeScript repository or the
# standalone Python repository without requiring an editable install first.
PACKAGE_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "sogni_client").is_dir()
)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
