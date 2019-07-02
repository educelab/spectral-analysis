import argparse

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    # im = cv2.imread(args.input, flags=cv2.IMREAD_ANYDEPTH)
    # im = cv2.imread(args.input)
    im = np.array(Image.open(args.input), dtype=np.float32)
    # cv2.imshow('image', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # im = im.astype(int)
    # im = cv2.equalizeHist(im)
    cv2.imwrite(args.output, im)


if __name__ == '__main__':
    main()
