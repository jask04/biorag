"""Download BEIR NFCorpus and write the normalized local form.

Run from the project root:

    uv run python scripts/build_corpus.py

Output goes to ``data/nfcorpus/`` (gitignored).
"""

from __future__ import annotations

from biorag.corpus import DATA_DIR, download_nfcorpus


def main() -> None:
    print(f"Downloading BEIR NFCorpus → {DATA_DIR}")
    stats = download_nfcorpus(DATA_DIR)
    print(stats.render())


if __name__ == "__main__":
    main()
