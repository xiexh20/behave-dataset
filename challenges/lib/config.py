"""
some dataset configurations
"""
import numpy as np

# camera intrinsic
BEHAVE_CAM_K = np.array([[979.784, 0, 1018.952],
                         [0, 979.840, 779.486],
                         [0, 0, 1.0]])
ICAP_CAM_K = np.array([[912.862, 0, 956.720],
                       [0, 912.676, 554.216],
                       [0, 0, 1.]])
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
DATASET_NAMES=['behave', 'icap', 'synz']