from wmr.base import Removal
import numpy as np

from wmr.attack_utils import remove_noise_fft_rgb, remove_noise_wavelet_rgb


class FilterRemoval(Removal):
    """
        Low-pass filter on image
    """
    def __init__(self, args):
        super().__init__(args)
    
    def _remove_watermark(self, original_image):
        # FFT-based low-pass filter
        removals = []
        removals.append(remove_noise_fft_rgb(original_image, cutoff=50))
        removals.append(remove_noise_wavelet_rgb(original_image, level=1))

        return np.array(removals)
