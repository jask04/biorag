"""Assemble a Hugging Face Space deploy directory from this repo.

HF Spaces installs ``requirements.txt`` and runs ``streamlit run app.py``
from the repo root, so the ``biorag`` package must sit at the root (not
under ``src/``). This script stages a self-contained Space tree:

    <out>/
      README.md          (front-matter, from deploy/space_readme.md)
      Dockerfile         (runs Streamlit; from deploy/Dockerfile)
      app.py
      requirements.txt
      biorag/            (copied from src/biorag)
      eval_results/      (the benchmark JSONs the Benchmark tab reads)

Then push ``<out>`` to the Space's git remote. See DEPLOY.md.

    uv run python scripts/prepare_space.py --out .hf_space
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare(out: Path) -> None:
    # Clear prior staged content but preserve a ``.git`` directory if the
    # Space remote has already been set up here — so redeploys are just
    # ``prepare_space`` + ``git add/commit/push`` with no re-init.
    if out.exists():
        for child in out.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)

    # The app never imports biorag.eval, and that subpackage pulls the
    # ragas/langchain deps deliberately excluded from the Space image —
    # so skip it (and bytecode caches) for a clean, minimal tree.
    shutil.copytree(
        ROOT / "src" / "biorag",
        out / "biorag",
        ignore=shutil.ignore_patterns("__pycache__", "eval"),
    )
    shutil.copy2(ROOT / "app.py", out / "app.py")
    shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")
    shutil.copy2(ROOT / "deploy" / "Dockerfile", out / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "space_readme.md", out / "README.md")

    results_src = ROOT / "eval_results"
    if results_src.exists():
        shutil.copytree(results_src, out / "eval_results")

    print(f"Staged Space tree at {out}")
    for item in sorted(out.rglob("*")):
        if item.is_file():
            print(f"  {item.relative_to(out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".hf_space", type=Path)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
