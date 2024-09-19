import numpy as np
from PIL import Image

from wmr.attack_utils import average_images, introduce_discontinuity

class Removal:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
    
    def _remove_watermark(self, original_image):
        raise NotImplementedError("Subclasses must implement this method")

    def remove_watermark(self, original_image):
        cleansed_image = self._remove_watermark(original_image)

        # Aggregate, if multiple copies provided
        if len(cleansed_image.shape) > 3:
            if self.args.aggregation == "mean":
                cleansed_image = average_images(cleansed_image)
            elif self.args.aggregation == "random":
                cleansed_image = introduce_discontinuity(cleansed_image, block_size=8)
            else:
                raise NotImplementedError(f"Aggregation method {self.args.aggregation} not implemented")

        # Clip in (0, 255) range
        cleansed_image = np.clip(cleansed_image, 0., 255.)
        # Make sure image is in the correct format
        cleansed_image = cleansed_image.astype(np.uint8)
        # Convert the averaged array back to an image
        cleansed_image = Image.fromarray(cleansed_image)
        return cleansed_image
