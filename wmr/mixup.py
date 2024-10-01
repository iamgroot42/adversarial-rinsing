import numpy as np
import torch as ch

from wmr.config import ExperimentConfig
from wmr.base import Removal
import kornia.filters as filters
import kornia.geometry.transform as geo
import kornia.enhance as enhance
from PIL import Image


class Mixup(Removal):
    """
        Add a random image to the original image
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # Load up rickroll image
        path = "/home/groot/work/erasing-the-invisible/Rickroll.jpg"
        # Load as numpy image
        self.rickroll = np.array(Image.open(path))/ 255.
        # Crop to 512x512
        self.rickroll = self.rickroll[0:512, 0:512]

    def _remove_watermark(self, original_image):
        image_np = np.array(original_image) / 255.

        rickroll_ratio = 0.08
        combined_image = rickroll_ratio * self.rickroll + (1 - rickroll_ratio) * image_np

        # Make sure the image is in the right range
        combined_image = np.clip(combined_image, 0, 1) * 255
        
        return combined_image
