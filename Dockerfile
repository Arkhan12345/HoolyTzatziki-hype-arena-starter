FROM python:3.12.16-slim-trixie

# No home and a read-only root at run time: every cache points at /tmp.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    HF_HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    UNIVERSE_DIR=/universe \
    OUTPUT_DIR=/output

WORKDIR /srv

# libgomp is LightGBM's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY app/ app/
COPY models/ models/
COPY data/ data/
COPY contracts/ contracts/

CMD ["python", "-m", "app.run"]

USER 10001
