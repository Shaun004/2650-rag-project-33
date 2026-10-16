#!/usr/bin/env python3
"""Print official Samsung download instructions (no scraping)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"


def main() -> None:
    print("Download manuals only from official Samsung support pages.\n")
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Model: {data.get('model_number')}")
        print(f"  Support URL: {data.get('document_url')}")
        print(f"  Place PDF at: {data.get('local_pdf')}")
        print(f"  Text used by pipeline: {data.get('local_text')}")
        print()
    print("After downloading a new PDF/text, add/edit a manifest and run:")
    print("  python scripts/ingest_manuals.py")


if __name__ == "__main__":
    main()
