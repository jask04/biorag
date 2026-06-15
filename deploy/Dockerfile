# Hugging Face Space (Docker SDK) image for the biorag Streamlit app.
# HF no longer offers a one-click Streamlit SDK, so we run Streamlit under
# the Docker SDK — which also pins Python 3.12 and the listen port.
FROM python:3.12-slim

# HF Spaces run the container as uid 1000; create a matching user so the
# app dir, the corpus it downloads to data/, and the HF model cache are all
# writable at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
