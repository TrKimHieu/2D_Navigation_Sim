# hm3denv image: the environments, the hm3d CLI and the numba kernels.
# Datasets and caches are mounted at run time, never baked into the image.
#
#   docker build -t hm3denv .                                   # environments + CLI
#   docker build -t hm3denv:build --build-arg EXTRAS=fast,build .   # + dataset building tools
#   docker build --target test .                                # runs the test suite
#
#   docker run --rm -v /data/hm3d/datasets:/data/datasets:ro -v hm3d-cache:/cache \
#       hm3denv bench svg-v1 --robot turtlebot4 --num-envs 1 8 32

ARG BASE=python:3.10-slim
FROM ${BASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    HM3D_DATASETS=/data/datasets HM3D_CACHE=/cache NUMBA_CACHE_DIR=/cache/numba

WORKDIR /opt/hm3denv
COPY pyproject.toml README.md ./
COPY src ./src
ARG EXTRAS=fast
RUN pip install ".[${EXTRAS}]"

RUN mkdir -p /data/datasets /cache/numba && chmod -R a+rwX /cache
VOLUME ["/cache"]
ENTRYPOINT ["hm3d"]
CMD ["datasets"]

# ---------------------------------------------------------------- tests
FROM runtime AS test
RUN pip install ".[build,test,fast]"
COPY tests ./tests
RUN HM3D_CACHE=/tmp/hm3d-cache python -m pytest -q -m "not workspace" -p no:cacheprovider
