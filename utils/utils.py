#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import os


def get_class_weight(labels_dir, class_cmap):
    """
    Arguments:
        labels_dir: directory where the image segmentation labels argparse
        class_cmap: dict style cmap (e.g. class: [r, g, b])

    Returns:
        class_weights: class balancing weights based on median frequency
    """
    image_files = [os.path.join(labels_dir, file)
                   for file in os.listdir(labels_dir)]

    class_pixels = np.zeros(len(class_cmap))
    total_pixels = np.zeros(len(class_cmap))

    cmap_keys = class_cmap.keys()

    for fn in image_files:
        image = cv2.imread(fn, 1)
        image_size = image.shape[0] * image.shape[1]

        for idx, cmap_key in enumerate(cmap_keys):
            color = np.array(class_cmap[cmap_key])
            class_map = np.all(np.equal(image, color), axis=-1)

            if np.sum(class_map) > 0.0:
                class_pixels[idx] += np.sum(class_map)
                total_pixels[idx] += image_size

    class_pixels = class_pixels.astype(np.float32)
    total_pixels = total_pixels.astype(np.float32)

    class_frequencies = np.true_divide(class_pixels, total_pixels)
    print("c_f: ", class_frequencies)
    meddian_frequency = np.median(class_frequencies)
    class_weights = meddian_frequency / class_frequencies

    return class_weights
