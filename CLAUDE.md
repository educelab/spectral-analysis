# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`spectral-analysis` (EduceLab, University of Kentucky) provides tools for processing and analyzing
multispectral image sets — primarily MegaVision captures of manuscripts. The installable package is
`spec_tools`; it ships three console scripts declared in `setup.cfg`.

## Commands

```bash
# Install (editable) into a venv
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install --editable .

# Run the full test suite (unittest, not pytest)
python -m unittest -v tests.test_spec_tools

# Run a single test
python -m unittest -v tests.test_spec_tools.UtilsTest.test_setup_logging

# Smoke check that entry points resolve (this is what CI does before tests)
spec-enhance -h
```

CI is GitHub Actions (`.github/workflows/`), on `main` pushes, `v*` tags, PRs, and manual dispatch:

- `ci.yml` — installs ExifTool via apt (`libimage-exiftool-perl`, *not* the bare `exiftool` package,
  which only exists on Debian), then `pip install .`, smoke-tests all three console scripts, then runs
  the unittest module against Python 3.12 and 3.13. There is no linter or formatter configured.
- `build_docker.yml` — builds the root `Dockerfile` for `linux/amd64,linux/arm64` (arm64 under QEMU
  emulation, so the job is slow) and pushes to `ghcr.io/educelab/spectral-analysis`. Pull requests
  build both platforms but do **not** push, as a Dockerfile-breakage check — so a PR validates exactly
  what a publish will build. It runs independently of `ci.yml` and does **not** gate on tests passing.

**The package is Python 3.12+** (`python_requires`). This is forced by the dependency stack, not
preference: `tifffile` requires `>=3.12` and numpy 2.5 dropped cp310/cp311, so older interpreters
cannot install current versions of the project's own dependencies. Keep `python_requires`, the CI
matrix, the README requirements list, and the `Dockerfile` base in sync when this moves.

Pin considerations for the image live in the `Dockerfile` header: the base is `python:3.13-slim`,
chosen because every compiled dependency (numpy, scipy, imagecodecs, scikit-image, scikit-learn,
pillow) has manylinux wheels for both arches there — `imagecodecs` ships `cp312-abi3`, whose stable
ABI is what covers 3.13+. Re-verify that before bumping, or the image needs a build toolchain.
ExifTool is installed from source and deliberately unpinned — `exiftool.org` only hosts the current
tarball, so pinned URLs 404 once a new version ships; SourceForge is the primary mirror with a
`gzip -t` validation guard.

**ExifTool 12+ must be on PATH** — `spec-enhance` shells out to it via `educelab.imgproc.exiftool` to
copy metadata tags to outputs.

## Architecture

Three layers, in dependency order:

- `spec_tools/utils/apps.py` — shared CLI helpers: `setup_logging`, `expand_path_list` (glob/dir
  expansion), `parse_roi_params` (parses ImageMagick-style `WxH+X+Y` into an `ROI` dataclass),
  `to_numpy_dtype` (maps the string `'8'`/`'16'`/`'32'` to `uint8`/`uint16`/`float32`, used directly
  as an argparse `type=`).
- `spec_tools/pca.py` — the only algorithm module, despite the name it covers both decomposition
  methods in `METHODS` (`'pca'` → `PCA`/`IncrementalPCA`, `'ica'` → `FastICA`). `fit(x, ...)` and
  `apply_transform(x, model)` take a stacked `(n_bands, H, W)` array, flatten to
  `(n_pixels, n_bands)` for scikit-learn, and reshape results back to per-component images.
  `method_of(model)` recovers the method name from a fitted estimator (used to label outputs when a
  pickled model is loaded). Component counts are read off `model.components_.shape[0]`, not
  `n_components_`, since `FastICA` lacks the latter. `fit` calls `sys.exit(1)` on invalid arguments
  rather than raising — it is written for CLI use, not as a library API.
- `spec_tools/apps/*.py` — each module is a self-contained `main()` bound to a console script:
  `enhance` → `spec-enhance`, `pca` → `spec-pca`, `apply_flats` → `spec-apply-flats`.

### External EduceLab dependencies

Most pixel work is delegated to the sibling packages `educelab-imgproc` and `educelab-cmdparse`, not
implemented here. Key touchpoints:

- `imgproc.pipeline.add_parser_enhancement_group(parser)` + `pipeline.parse_and_build(args.commands)`
  build the entire enhancement chain for `spec-enhance`. Because commands are parsed as a trailing
  list, users must separate them with `--`:
  `spec-enhance -i foo.tif -- -gamma -pstretch=2,98`. Available ops (gamma, clahe, stretch,
  pstretch, curves, sharpen, shadows-highlights, …) are defined in `imgproc/pipeline.py` — read that
  file rather than guessing at supported flags.
- `imgproc.as_dtype` / `imgproc.normalize` handle range-safe dtype conversion; `imgproc.flatfield_correction`
  backs `spec-apply-flats`.

### App conventions

`spec-enhance` and `spec-apply-flats` share a common shape worth preserving when adding apps: read
image → `img_as_float` → process → pick an output dtype (forced to `uint8` for bmp/jpg, else
`--output-depth`, else the input dtype) → apply format-specific write kwargs (jpeg `quality`, tiff
`zlib` compression) → write. `spec-enhance` additionally writes a timestamped JSON run log
(args + resolved pipeline) into the output directory and stamps a `-Software=` ExifTool tag.

Both the run log's `program` field and the `-Software=` tag report `spec_tools.__version__`, which
`spec_tools/__init__.py` derives from the installed package metadata via
`importlib.metadata.version('spectral-analysis')`. So the single source of truth is `version` in
`setup.cfg` — there is no per-app version constant to bump. If the package is not installed at all,
`__version__` falls back to `'0.0.0+unknown'` rather than asserting a version.

`spec-pca` selects its decomposition with `--method/-m` (`pca` default, or `ica`); `--incremental`
is PCA-only and errors out if combined with `ica`. It supports fit-then-persist workflows:
`--save-model`/`--load-model` pickle the sklearn estimator, and `--images-to-transform` applies a
model fit on one set to a different set. Note that when transforming a separate set, the app
requires the file count to equal the component count. Output files are prefixed with the method
that actually produced them (`pca_00.tif`, `ica_00.tif`) — derived from the loaded model, not
`--method`, so a loaded ICA pickle still yields `ica_` — unless `--output-prefix` overrides it.

## Other directories

- `scripts/` — standalone research/demo scripts, not installed and not tested. `convert_to_zarr.py` /
  `dump_zarr.py` demonstrate moving flatfield-corrected MegaVision datasets to Zarr (requires `zarr`
  and `numcodecs`, which are *not* in `requirements.txt`). `false_color_pipeline.py` encodes the
  lab's named PCA band groupings (A–F, filtered by UV/IR filter names in filenames) and false-color
  channel combinations (A–L); it calls `spec_tools.pca` directly, so signature changes there need to
  be mirrored here.
- `legacy/` — the pre-2.0 `spectral_analysis` package and experiment code (autoencoder/LLE/neural-net
  contrast enhancement). Excluded from the installed package; treat as reference only.
The project previously shipped a `singularity/` directory (an Apptainer definition plus a SLURM
`submit_job.sh`) that installed the source editable and expected a writable overlay for in-container
edits. Both were retired in favor of the Docker image; on the cluster, pull it with
`apptainer pull docker://ghcr.io/educelab/spectral-analysis:latest`. The image installs the package
non-editable and sets `WORKDIR /data`, so there is no overlay-based dev workflow — the SBATCH header
and rclone bind settings from `submit_job.sh` are recoverable from git history if needed.
