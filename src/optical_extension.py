import cv2
import numpy as np
from .localizer import DriftSenseLocalizer

class OpticalImageLocalizer:
    """
    Bonus Extension: Optical Microscope (RGB 3-Channel) Navigation Error Recovery.
    Converts 3-channel optical color images using Luminance & Color Contrast prior,
    then applies multi-scale correlation matching.
    """
    def __init__(self, nominal_scale=10.0):
        self.base_localizer = DriftSenseLocalizer(nominal_scale=nominal_scale)

    def preprocess_optical_rgb(self, rgb_img):
        """
        Converts RGB optical microscope image to single-channel enhanced grayscale representation.
        Uses LAB color space luminance (L) channel with CLAHE local contrast enhancement.
        """
        if len(rgb_img.shape) == 2:
            return rgb_img
        
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        return enhanced_l

    def localize_optical(self, ref_rgb, search_rgb):
        """
        Runs localization on RGB optical reference and search image pairs.
        """
        ref_mono = self.preprocess_optical_rgb(ref_rgb)
        search_mono = self.preprocess_optical_rgb(search_rgb)

        return self.base_localizer.localize(ref_mono, search_mono)
