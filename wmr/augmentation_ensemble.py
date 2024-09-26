import numpy as np
import torch as ch

from wmr.config import ExperimentConfig
from wmr.base import Removal
import kornia.filters as filters
import kornia.geometry.transform as geo
import kornia.enhance as enhance


class FilterEnsemble(Removal):
    """
        Similar logic to adversarial- add perturbations to images until exiting watermark detection models are fooled.
        Alternatively, could target watermark-extraction methods directly, or perhaps joint optimization for both?
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
    
    def _gaussian_blur(self, img):
        aug = filters.gaussian_blur2d(img, (3, 3), (1.5, 1.5))
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()

    def _motion_blur(self, img):
        aug =  filters.motion_blur(img, 5, 90., 1)
        # Convert to numpy (channels and all)
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()
    
    def _rotation(self, imf):
        aug = geo.rotate(imf, ch.tensor([2.]))
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()
    
    def _brightness(self, img):
        # in [0, 1]
        aug = enhance.adjust_brightness(img, 0.1)
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()
    
    def _contrast(self, img):
        # in [0, 1]
        aug = enhance.adjust_contrast(img, 1.1)
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()

    def _hue(self, img):
        # in [-pi, pi]
        aug = enhance.adjust_hue(img, np.pi * 0.1)
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()

    def _saturation(self, img):
        # in [0, 2]
        aug = enhance.adjust_saturation(img, 1.1)
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()
    
    def _posterize(self, img):
        # in [0, 8]
        aug = enhance.posterize(img, 5)
        return aug.permute(0, 2, 3, 1).squeeze(0).numpy()

    def _remove_watermark(self, original_image):
        augmented = []
        image_ch = ch.tensor(np.array(original_image)).unsqueeze(0).permute(0, 3, 1, 2)
        image_ch = image_ch.float() / 255.

        augmented.append(self._gaussian_blur(image_ch))
        augmented.append(self._motion_blur(image_ch))
        # augmented.append(self._rotation(image_ch))
        augmented.append(self._brightness(image_ch))
        augmented.append(self._contrast(image_ch))
        augmented.append(self._hue(image_ch))
        augmented.append(self._saturation(image_ch))
        augmented.append(self._posterize(image_ch))
        
        all_together = np.array(augmented) * 255
        return all_together
