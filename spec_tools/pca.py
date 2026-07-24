import logging
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA, FastICA

# Supported decomposition methods
METHODS = ('pca', 'ica')


def method_of(model) -> str:
    """Return the method name which produced a fitted model"""
    if isinstance(model, FastICA):
        return 'ica'
    elif isinstance(model, (IncrementalPCA, PCA)):
        return 'pca'
    else:
        raise TypeError(f'unrecognized model type: {type(model).__name__}')


def fit(x, components: int = None, batch_size: int = None,
        incremental: bool = False, method: str = 'pca', roi=None):
    logger = logging.getLogger(__name__)
    # Validate the decomposition method
    if method not in METHODS:
        logger.error(f'Unrecognized method ({method}). Supported methods: '
                     f'{", ".join(METHODS)}')
        sys.exit(1)

    # Validate number of components
    if components is not None and not (1 <= components < x.shape[0]):
        logger.error(f'Requested components ({components}) outside '
                     f'range [1, {x.shape[0]}]')
        sys.exit(1)

    # Setup new model
    if method == 'ica':
        if incremental:
            logger.error('Incremental fitting is not supported for ICA')
            sys.exit(1)
        model = FastICA(n_components=components, random_state=42)
    elif incremental:
        model = IncrementalPCA(n_components=components,
                               batch_size=batch_size)
    else:
        model = PCA(n_components=components)

    # Crop training data to ROI
    if roi is not None:
        logging.debug(f'Using input ROI: {roi}')
        x = x[:, roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]

    # Flatten input
    x_flat = x.reshape((x.shape[0], -1))
    x_flat = np.swapaxes(x_flat, 0, 1)

    # Fit input files
    logger.debug(f'Fitting {x.shape[0]} images')
    model.fit(x_flat)
    return model


def apply_transform(x, model):
    logger = logging.getLogger(__name__)
    # Transform images
    logger.debug(f'Transforming {x.shape[0]} images')
    x_flat = x.reshape((x.shape[0], -1))
    x_flat = np.swapaxes(x_flat, 0, 1)
    x_flat = model.transform(x_flat)

    # Convert back to images
    x_flat = np.swapaxes(x_flat, 0, 1)
    out_shape = (model.components_.shape[0],) + x.shape[1:]
    return x_flat.reshape(out_shape)
