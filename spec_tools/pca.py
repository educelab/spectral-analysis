import logging
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA, FastICA


def fit(x, components: int = None, batch_size: int = None,
        incremental: bool = False, roi=None):
    logger = logging.getLogger(__name__)
    # Validate number of components
    if components is not None and not (1 <= components < x.shape[0]):
        logger.error(f'Requested components ({components}) outside '
                     f'range [1, {x.shape[0]}]')
        sys.exit(1)

    # Setup new PCA
    if incremental:
        pca = IncrementalPCA(n_components=components,
                             batch_size=batch_size)
    else:
        pca = FastICA(n_components=components, random_state=42)

    # Crop training data to ROI
    if roi is not None:
        logging.debug(f'Using input ROI: {roi}')
        x = x[:, roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]

    # Flatten input
    x_flat = x.reshape((x.shape[0], -1))
    x_flat = np.swapaxes(x_flat, 0, 1)

    # Fit input files
    logger.debug(f'Fitting {x.shape[0]} images')
    pca.fit(x_flat)
    return pca


def apply_transform(x, pca):
    logger = logging.getLogger(__name__)
    # Transform images
    logger.debug(f'Transforming {x.shape[0]} images')
    x_flat = x.reshape((x.shape[0], -1))
    x_flat = np.swapaxes(x_flat, 0, 1)
    x_flat = pca.transform(x_flat)

    # Convert back to images
    x_flat = np.swapaxes(x_flat, 0, 1)
    pca_shape = (pca.components_.shape[0],) + x.shape[1:]
    return x_flat.reshape(pca_shape)
