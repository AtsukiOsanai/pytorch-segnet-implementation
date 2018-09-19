#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import yaml


class SeglabelTransformer():
    """
    This class has functions which encodes colored image into seglabeled image
    and decodes seglabeled image into mask image.
    """
    def __init__(self, config, alpha=0.3):
        self.class_number = config["class_number"]
        self.class_cmap = config["class_cmap"]
        self.class_keys = self.class_cmap.keys()
        self.alpha = alpha

    def encode(self, image):
        """
        - input
            image: np.array (W, H, C)
        - output
            label: np.array (H, W)
        """
        h, w = image.shape[:2]
        label = np.zeros((h, w))

        for class_idx, key in enumerate(self.class_keys):
            color = np.array(self.class_cmap[key])
            mask = cv2.inRange(image, color, color)
            mask_indices = np.where(mask == 255)
            label[mask_indices] = class_idx + 1

        return label

    def decode(self, label):
        """
        - input
            label: np.array (W, H)
        - output
            output: np.array (W, H, 3)
        """
        w, h = label.shape[:2]
        output = np.zeros((w, h, 3))

        for class_idx, key in enumerate(self.class_keys):
            mask_indices = np.where(label == class_idx + 1)
            output[mask_indices] = self.class_cmap[key]

        output = output.astype(np.uint8)

        return output

    def overlay(self, image, label):
        """
        - input
            image: np.array (W, H, C)
            label: np.array (W, H, C)
        -output
            overlay: np.array (W, H, C)
        """
        overlay = cv2.addWeighted(image, (1.0 - self.alpha),
                                  label, self.alpha,
                                  0.0)

        return overlay
