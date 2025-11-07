"""Download a Telugu language corpus from Wikipedia random articles.

The script fetches random Telugu Wikipedia pages using the public API and
saves their plaintext extracts into `data/telugu_corpus.txt`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import requests


API_URL = "https://te.wikipedia.org/w/api.php"


def fetch_random_telugu_pages(count: int, delay: float = 0.1) -> Iterable[str]:
    """Yield plaintext extracts for random Telugu Wikipedia pages."""

    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 1,
        "prop": "extracts",
        "explaintext": 1,
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; TeluguBPEBot/1.0; "
            "+https://example.org/contact)"
        )
    }

    for _ in range(count):
        try:
            response = requests.get(
                API_URL, params=params, headers=headers, timeout=15
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, json.JSONDecodeError):
            time.sleep(delay)
            continue

        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            time.sleep(delay)
            continue

        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()
        if extract:
            yield extract

        time.sleep(delay)


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "telugu_corpus.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    texts = list(fetch_random_telugu_pages(400))
    if not texts:
        raise RuntimeError("Failed to download any Telugu Wikipedia pages.")

    merged_text = "\n\n".join(texts)
    output_path.write_text(merged_text, encoding="utf-8")
    print(f"Saved {len(texts)} articles to {output_path}.")


if __name__ == "__main__":
    main()

