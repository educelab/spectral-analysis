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

CI (`.gitlab-ci.yml`) installs ExifTool via apt, then `pip install .`, then `spec-enhance -h` and the
unittest module, against Python 3.10 and 3.11. There is no linter or formatter configured.

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
(args + resolved pipeline) into the output directory and stamps a `-Software=` ExifTool tag; bump
`ENHANCE_VERSION` in `spec_tools/apps/enhance.py` when its behavior changes.

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
- `singularity/` — Apptainer/Singularity definition used on the UK HPC cluster, plus a SLURM
  `submit_job.sh`. The container installs the source editable at
  `/usr/local/educelab/spectral-analysis` and expects to be run with a writable overlay so users can
  check out branches inside the container.
