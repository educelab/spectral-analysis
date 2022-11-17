import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

SHAPE_REGEX = r"(?P<w>\d+)x(?P<h>\d+)"
ORIGIN_REGEX = r"\+(?P<x>\d+)\+(?P<y>\d+)"
ROI_REGEX = SHAPE_REGEX + ORIGIN_REGEX
SHAPE_REGEX = re.compile(SHAPE_REGEX)
ORIGIN_REGEX = re.compile(ORIGIN_REGEX)
ROI_REGEX = re.compile(ROI_REGEX)


def setup_logging(log_level: int = logging.INFO):
    logging.basicConfig(
        format='[%(name)s] %(levelname)s: %(message)s',
        level=log_level
    )


def expand_path_list(
        files: Union[str | List[str | os.PathLike]],
        recursive: bool = False):
    logger = logging.getLogger(__name__)

    # Promote to list
    if isinstance(files, str):
        files = [files]

    # Construct final files list
    final_files = []
    for file in files:
        path = Path(file)
        if path.is_file():
            final_files.append(path)
        elif path.is_dir():
            for s in path.iterdir():
                if s.is_file():
                    final_files.append(Path(s))
                elif s.is_dir() and recursive:
                    final_files.extend(expand_path_list([s], recursive))
        else:
            logger.warning(f'Skipping: {file}')
    return final_files


def parse_roi_params(roi_string: str):
    logger = logging.getLogger(__name__)

    # ROI return value
    @dataclass
    class ROI:
        x: int = None
        y: int = None
        w: int = None
        h: int = None

        def __str__(self):
            return f'(x:{self.x}, y:{self.y}, w:{self.w}, h:{self.h})'

    # Parse the ROI parameters
    match = ROI_REGEX.match(roi_string)
    if not match:
        logger.warning(f'Warning: Cannot parse ROI argument: {roi_string}. '
                       f'Ignoring.')
        return ROI()

    # Convert to ints
    roi = match.groupdict()
    convert_error = False
    for key, value in roi.items():
        roi[key] = int(value)

    if convert_error:
        return ROI()

    # Return ROI commands
    return ROI(roi['x'], roi['y'], roi['w'], roi['h'])


def to_numpy_dtype(v: str) -> np.ScalarType:
    """Transform a numeric string to a numpy dtype"""
    if v == '8':
        return np.uint8
    elif v == '16':
        return np.uint16
    elif v == '32':
        return np.float32
    else:
        raise ValueError(f'depth {v} not recognized/supported')
