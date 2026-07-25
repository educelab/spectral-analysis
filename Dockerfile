# syntax=docker/dockerfile:1
#
# spectral-analysis runtime image. Bundles ExifTool (required by spec-enhance,
# which shells out to it via educelab.imgproc.exiftool) plus the spec_tools
# package and its three console scripts.
#
# Replaces the retired singularity/spectral-analysis.def. On the HPC cluster,
# pull it directly instead of building a .sif from source:
#   apptainer pull docker://ghcr.io/educelab/spectral-analysis:latest
#
# Multi-arch (linux/amd64, linux/arm64): every compiled dependency (numpy,
# imagecodecs, scikit-image, scikit-learn) ships manylinux wheels for both arches
# at 3.12+, so no build toolchain is needed here. 3.12 is the floor for that --
# numpy 2.5 dropped cp311, and imagecodecs' wheels are cp312-abi3 (stable ABI,
# so they also cover 3.13+). 3.13 works too; verify wheel coverage for all four
# compiled deps on both arches before bumping, or this image needs gcc/g++/make.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="spectral-analysis"
LABEL org.opencontainers.image.description="Tools for processing and analyzing multispectral image sets."
LABEL org.opencontainers.image.authors="Seth Parker <c.seth.parker@uky.edu>"
LABEL org.opencontainers.image.source="https://github.com/educelab/spectral-analysis"
LABEL org.opencontainers.image.url="https://github.com/educelab/spectral-analysis"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# curl + ca-certificates -> fetch the ExifTool tarball
# make + perl            -> build and install ExifTool from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        make \
        perl \
    && rm -rf /var/lib/apt/lists/*

# ExifTool (12+ required).
#
# Deliberately unpinned: exiftool.org only ever hosts the *current* tarball, so
# a pinned URL 404s as soon as a new version ships -- which is why 459929d moved
# the retired .def to ver.txt. Distribution is really through SourceForge, so
# prefer that mirror and fall back to exiftool.org. The SourceForge redirector
# can hand back an HTML error page instead of the tarball, so validate the
# download with `gzip -t` before extracting. Mirrors pgs-recon's Dockerfile.
RUN set -eux \
    && mkdir -p /usr/local/educelab/exiftool/ \
    && cd /usr/local/educelab/exiftool/ \
    && EXIFTOOL_VER="$(curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://exiftool.sourceforge.net/ver.txt \
        || curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://exiftool.org/ver.txt)" \
    && echo "Installing ExifTool ${EXIFTOOL_VER}" \
    && TARBALL="Image-ExifTool-${EXIFTOOL_VER}.tar.gz" \
    && ( curl -fSL --retry 5 --retry-delay 3 --retry-all-errors -o "${TARBALL}" \
            "https://downloads.sourceforge.net/project/exiftool/${TARBALL}" \
         || curl -fSL --retry 5 --retry-delay 3 --retry-all-errors -o "${TARBALL}" \
            "https://exiftool.org/${TARBALL}" ) \
    && gzip -t "${TARBALL}" \
    && tar -xzf "${TARBALL}" \
    && cd "Image-ExifTool-${EXIFTOOL_VER}/" \
    && perl Makefile.PL \
    && make test \
    && make install \
    && cd / \
    && rm -rf /usr/local/educelab/exiftool \
    && exiftool -ver

# Install the package. Non-editable: unlike the retired .def, this image is not
# meant to be edited in place via a writable overlay.
COPY . /usr/local/educelab/spectral-analysis
RUN python3 -m pip install --upgrade pip wheel setuptools \
    && python3 -m pip install /usr/local/educelab/spectral-analysis

# These are batch CLIs taking input/output paths, so default to a mount point
# rather than the source tree:
#   docker run --rm -v "$PWD:/data" <image> spec-pca -i 'training_set/*.tif'
WORKDIR /data

# No ENTRYPOINT: any console script (spec-enhance, spec-pca, spec-apply-flats)
# can be used as the command, and Apptainer's generated runscript execs passed
# args directly. Bare `docker run <image>` prints spec-enhance's help.
CMD ["spec-enhance", "-h"]
