from PIL import Image
import imageio
import pyheif
import io
import numpy as np

from wmr.base import Removal


# Function to compress the image as JPEG
def compress_jpeg(image, quality: int=85):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# Function to compress the image as WebP
def compress_webp(image, quality: int=85):
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


class CompressionRemoval(Removal):
    def __init__(self, args):
        super().__init__(args)
    
    def remove_watermark(self, original_image):
        quality = 75  # Default quality for compression

        def image_to_array(image):
            return np.array(image)

        # Compress the image using different methods
        jpeg_compressed = compress_jpeg(original_image, quality=quality)
        webp_compressed = compress_webp(original_image, quality=quality)
        # Add more compression formats if needed

        # Convert images to arrays
        jpeg_array = image_to_array(jpeg_compressed)
        webp_array = image_to_array(webp_compressed)

        # Ensure all images are of the same size for averaging
        if jpeg_array.shape != webp_array.shape:
            webp_array = np.resize(webp_array, jpeg_array.shape)

        # Average the pixel values across the images
        all_together = np.stack([jpeg_array.astype(np.float32), webp_array.astype(np.float32)])
        average_image_array = np.mean(all_together, axis=0)
        average_image_array = average_image_array.astype(np.uint8)

        # Convert the averaged array back to an image
        average_image = Image.fromarray(average_image_array)

        return average_image
