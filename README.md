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
git clone https://gitlab.com/educelab/spectral-analysis.git

# (optional) setup a virtual environment
pyhon3 -m venv spectral-analysis/venv/
source spectral-analysis/venv/bin/activate

# install the project
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install spectral-analysis/
```

## API
The `spec_tools` module is currently in active development and the API is not 
stable.


## Apps
### spec-enhance
Apply contrast enhancement to input images:

```bash
# Run simple image normalization. 
# Similar to ImageMagick's -auto-level flag.
spec-enhance -i foo01.tif foo02.tif bar*.tif -o normalized/

# Run contrast enhancement with gamma correction and white/black level clipping.
# Similar to ImageMagick's -level flag.
spec-enhance -i foo01.tif foo02.tif bar*.tif -o enhanced/ --enhance

# Run CLAHE contrast enhancement. 
# Applied in addition to --enhance (if provided).
spec-enhance -i foo01.tif foo02.tif bar*.tif -o clahe/ --clahe
```

### spec-pca
Apply PCA transforms to a set of equally-sized images:

```bash
# Fit PCA to an image set and save all principal component images
spec-pca -i training_set/*.tif

# Fit PCA to an image set and save the top 5 principal component images
spec-pca -i training_set/*.tif -c 5

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
