# Deploying biorag to Hugging Face Spaces

The app runs on a free Hugging Face Space (Streamlit SDK). It reads its
vectors from your Qdrant Cloud cluster and self-bootstraps the NFCorpus
corpus on first boot, so the only manual setup is creating the Space and
its three secrets.

## 1. Create the Space

1. Sign in at <https://huggingface.co> and create an access token with
   **write** scope: Settings → Access Tokens → New token.
2. Create a new Space: <https://huggingface.co/new-space>
   - Owner: your account · Space name: `biorag`
   - SDK: **Docker** → **Blank** template (HF no longer offers a one-click
     Streamlit SDK; the bundled `Dockerfile` runs Streamlit)
   - Hardware: **CPU basic (free)** · Visibility: Public

## 2. Set the Space secrets

In the Space → Settings → *Variables and secrets*, add three **secrets**:

| Name | Value |
|---|---|
| `GOOGLE_API_KEY` | your Google AI Studio key (shared demo fallback) |
| `QDRANT_URL` | your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | your Qdrant Cloud API key |

The Qdrant cluster must already hold the general (`BGE-small`) index —
build it once locally with `uv run python scripts/build_index.py`.

## 3. Stage and push

The `biorag` package lives under `src/` for local development, but a Space
expects it at the repo root. `scripts/prepare_space.py` stages a
self-contained tree (package at the root, `Dockerfile`, `requirements.txt`,
the front-matter `README.md`, and the `eval_results/` JSONs for the
Benchmark tab):

```bash
uv run python scripts/prepare_space.py --out .hf_space

cd .hf_space
git init -b main
git remote add origin https://huggingface.co/spaces/<your-username>/biorag
git add .
git commit -m "Deploy biorag"
git push -f origin main      # authenticate with your HF write token
```

The Space builds, installs `requirements.txt`, downloads NFCorpus on first
boot (~30s), and serves the app.

**Redeploy after changes** — `prepare_space.py` preserves the `.git` it
finds in `.hf_space`, so subsequent deploys are just:

```bash
uv run python scripts/prepare_space.py --out .hf_space
cd .hf_space && git add . && git commit -m "Update" && git push
```

## Notes

- The Space image excludes the eval stack (`ragas`, `langchain*`) — those
  aren't imported by `app.py`, which keeps the build small and fast.
- Visitors without their own key share the Space's `GOOGLE_API_KEY` until
  its daily free-tier quota is reached, then the UI prompts them to paste
  their own key. Retrieval keeps working regardless.
