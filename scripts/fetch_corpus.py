"""Download the Fair Work Ombudsman fact sheets listed in rag/settings.py into data/raw/."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets `python scripts/fetch_corpus.py` import the rag package

from rag.settings import CORPUS_URLS  # noqa: E402

RAW = ROOT / "data" / "raw"


def download(url: str) -> bytes:
    """Fetch one URL and return its bytes, rejecting anything that is not a PDF."""
    request = urllib.request.Request(url, headers={"User-Agent": "au-workplace-assistant/0.1 (personal portfolio project; one-off corpus download)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    return data


def main() -> int:
    """Download every missing fact sheet, print one line per file, and return 1 if any download failed."""
    RAW.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, url in CORPUS_URLS.items():
        target = RAW / name
        if target.exists():
            print(f"skip    {name} (already present)")
            continue
        try:
            data = download(url)
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"FAILED  {name}: {exc}")
            continue
        target.write_bytes(data)
        print(f"ok      {name} ({len(data) // 1024} KB)")
    print(f"{len(CORPUS_URLS) - failed}/{len(CORPUS_URLS)} fact sheets in data/raw/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
