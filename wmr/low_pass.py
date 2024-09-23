from wmr.base import Removal
import numpy as np

from wmr.attack_utils import remove_noise_fft_rgb, remove_noise_wavelet_rgb
from wmr.config import ExperimentConfig


class FilterRemoval(Removal):
    """
        Low-pass filter on image
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
    
    def _remove_watermark(self, original_image):
        # FFT-based low-pass filter
        removals = []
        removals.append(remove_noise_fft_rgb(original_image, cutoff=100))
        removals.append(remove_noise_wavelet_rgb(original_image, level=2))

        return np.array(removals)
