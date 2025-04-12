import cv2
import numpy as np


def save_grayscale(path: str, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)
