# Spectral Analysis Tools

Tools for processing and analyzing spectral image sets.

[TOC]

## Requirements
- Python 3.10+
- [ExifTool 12+](https://exiftool.org/install.html)

## Installation

### From source
```bash
# get source code
git clone https://github.com/educelab/spectral-analysis.git

# (optional) setup a virtual environment
python3 -m venv spectral-analysis/venv/
source spectral-analysis/venv/bin/activate

# install the project
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install spectral-analysis/
```

### Docker
A prebuilt image is published to the GitHub Container Registry with ExifTool
already installed. Tags: `latest` (most recent release), `edge` (tip of `main`),
and one per release version (e.g. `2.1.0`).

```bash
docker pull ghcr.io/educelab/spectral-analysis:latest

# mount the current directory at /data (the image's working directory)
docker run --rm -v "$PWD:/data" ghcr.io/educelab/spectral-analysis \
  spec-pca -i 'training_set/*.tif' -o training_pca/
```

Quote glob patterns so they are expanded inside the container rather than by
your shell against host paths. Any of the three console scripts can be used as
the command.

### Apptainer/Singularity (HPC)
Pull the same image directly — there is no separate container definition to
build:

```bash
apptainer pull spectral-analysis.sif docker://ghcr.io/educelab/spectral-analysis:latest
apptainer run spectral-analysis.sif spec-pca -i 'training_set/*.tif' -o training_pca/
```

## API
The `spec_tools` module is currently in active development and the API is not 
stable.


## Apps
### spec-enhance
Apply contrast enhancement to input images:

```bash
# apply gamma correction using default (1./2.2) followed by contrast stretching to [2%, 98%] of histogram  
spec-enhance -i foo.tif -- -gamma -stretch=2,98

# apply CLAHE contrast enhancement followed by gamma correction
spec-enhance -i foo.tif -- -clahe -gamma

# apply gamma correction twice (why, though?)
spec-enhance -i foo.tif -- -gamma -gamma=2.2
```

### spec-pca
Apply dimensionality reduction transforms to a set of equally-sized images:

```bash
# Fit PCA to an image set and save all principal component images
spec-pca -i training_set/*.tif

# Fit PCA to an image set and save the top 5 principal component images
spec-pca -i training_set/*.tif -c 5

# Fit Independent Component Analysis (ICA) instead of PCA
spec-pca -i training_set/*.tif -m ica

# Fit PCA to an image set region-of-interest
spec-pca -i training_set/*.tif --roi 800x600+200+100

# Fit PCA to one image set and apply the model to a different set
spec-pca -i training_set/*.tif --images-to-transform inference_set/*.tif -o inference_pca/

# Fit PCA to one image set, then save and apply the model to a different set
spec-pca -i training_set/*.tif -o training_pca/ --save-model pca.pickle
spec-pca -i inference_set/*.tif -o inference_pca/ --load-model pca.pickle
```

## License
The software in this repository is licensed under the 
[GNU General Public License v3.0](LICENSE). This project is free software: you 
can redistribute it and/or modify it under the terms of the GPLv3 or (at your 
option) any later version.
