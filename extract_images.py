#!/usr/bin/env python

from keras.preprocessing import image
import numpy as np
import glob
from tqdm import tqdm
import os
import sys

TARGET_WIDTH = 224
TARGET_HEIGHT = 224


def image_id(filename):
    return int(os.path.splitext(
        os.path.basename(filename))[0])


def condense(dirname, out_filename):
    print('Extracting from {}'.format(dirname), file=sys.stderr)
    files = glob.glob(os.path.join(dirname, '*.jpg'))
    files = sorted(files, key=image_id)
    nfiles = len(files)

    out = np.zeros((nfiles, TARGET_WIDTH, TARGET_HEIGHT, 3))

    with tqdm(total=nfiles) as pbar:
        for (i, filename) in enumerate(files):
            img = image.load_img(filename, target_size=(TARGET_WIDTH, TARGET_HEIGHT))
            out[i] = image.img_to_array(img)
            pbar.update(1)

    np.save(out_filename, out)


def main():
    condense('data/train', 'train.npy')
    condense('data/test', 'test.npy')


if __name__ == '__main__':
    main()
