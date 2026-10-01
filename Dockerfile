FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ROBUSTLAB_DEVICE=cpu \
    ROBUSTLAB_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Hugging Face Spaces run the container as UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

# Only runtime code is copied: the training pipeline and local artifacts stay out of the image.
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user robustlab ./robustlab
COPY --chown=user app.py ./

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
# GPy imports matplotlib on start-up unless plotting is disabled; the Space never plots with GPy.
RUN mkdir -p /home/user/.config/GPy && printf "[plotting]\nlibrary = none\n" > /home/user/.config/GPy/user.cfg

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
