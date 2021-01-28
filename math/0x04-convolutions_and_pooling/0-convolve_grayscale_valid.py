#!/usr/bin/env python3
"""Module to perform a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.
    Arguments:
        images (numpy.ndarray): an array with shape (m, h, w)
                                containing multiple grayscale images,
                                where:
                                    - m is the number of images.
                                    - h is the height in pixels of the images.
                                    - w is the width in pixels of the images.
        kernel (numpy.ndarray): an array with shape (kh, kw)
                                containing the kernel for the convolution,
                                where:
                                    - kh is the height of the kernel.
                                    - kw is the width of the kernel.
    Returns:
        numpy.ndarray: an array containing the convolved images.
    """

    # Save Dimmensions
    kh, kw = kernel.shape
    m, h, w = images.shape
    # Make output template
    output_shape = (m, h - kh + 1, w - kw + 1)
    output = np.zeros(output_shape)
    # Fill template
    for row in range(output_shape[1]):
        for column in range(output_shape[2]):
            # Split the part that you need from every image
            sub_matrix = images[:, row: row + kh, column: column + kw]
            # Apply the kernel and sum every resultant matrix to get
            # the convolution value in that point for every image
            output[:, row, column] = (sub_matrix * kernel).sum(axis=(1, 2))

    return output
