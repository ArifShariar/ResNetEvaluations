import os.path

import cv2
import imageio.v2 as imageio


def clahe_opencv(image, clip_limit, tile_grid_size):
    """
    Enhances an image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    :param image: input image
    :param clip_limit: clip limit
    :param tile_grid_size: grid size
    :return: enhanced image
    """
    equalized_image = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl1 = clahe.apply(equalized_image)
    ret, thresh1 = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh1, cl1
