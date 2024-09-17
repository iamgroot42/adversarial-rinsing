from PIL import Image
import io
import numpy as np

from wmr.base import Removal


# Function to compress the image as JPEG
def compress_jpeg(image, quality: int=85):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer)).astype(np.float32)

# Function to compress the image as WebP
def compress_webp(image, quality: int=85):
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer)).astype(np.float32)

# Quantize image (reduce number of unique colors)
def quantize(image, num_colors: int = 256):
    quantized = image.quantize(colors=num_colors).convert("RGB")
    return np.asarray(quantized).astype(np.float32)

class CompressionRemoval(Removal):
    def __init__(self, args):
        super().__init__(args)
    
    def remove_watermark(self, original_image):
        quality = 75  # Default quality for compression

        quantization_color_coverage = 0.95
        # Out of curiosity, count how many colors the image has (unique)
        # Perhaps histogram instead?
        hist_counts = np.unique(np.asarray(original_image), return_counts=True)[1][::-1]
        cumulative_sum = np.cumsum(hist_counts)
        total_pixels = cumulative_sum[-1]
        cumulative_percentage = cumulative_sum / total_pixels
        # How many colors (unique) when included would cover 95% of all pixels?
        colors_needed = np.argmax(cumulative_percentage >= quantization_color_coverage) + 1

        # Compress the image using different methods
        compressed_images = []
        compressed_images.append(compress_jpeg(original_image, quality=quality))
        compressed_images.append(compress_webp(original_image, quality=quality))
        compressed_images.append(quantize(original_image, num_colors=colors_needed))

        # Average the pixel values across the images
        average_image_array = np.mean(np.stack(compressed_images), axis=0)
        average_image_array = average_image_array.astype(np.uint8)

        # Convert the averaged array back to an image
        average_image = Image.fromarray(average_image_array)

        return average_image
