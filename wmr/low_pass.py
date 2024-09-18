from wmr.base import Removal
from PIL import Image
import numpy as np
import pywt

def remove_noise_fft_rgb(image, cutoff: int=50):
    """
    Removes high-frequency noise from an RGB image using FFT.
    """
    image_np = np.array(image)

    # Split the image into R, G, B channels
    channels = [image_np[:, :, i] for i in range(3)]

    # Process each channel separately
    filtered_channels = []
    for channel in channels:
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)

        # Create a mask for the low-pass filter
        rows, cols = channel.shape
        crow, ccol = rows // 2 , cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1

        # Apply the mask and inverse FFT
        f_shifted_filtered = f_shifted * mask
        f_ishift = np.fft.ifftshift(f_shifted_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        filtered_channels.append(np.abs(image_filtered))

    # Recombine the filtered R, G, B channels
    image_filtered_np = np.stack(filtered_channels, axis=-1)
    return np.clip(image_filtered_np.astype(np.float32), 0., 255.)


def remove_noise_wavelet_rgb(image, wavelet: str='db1', level: int=1):
    """
        Removes noise from an RGB image using wavelet denoising.
    """
    image_np = np.array(image)

    # Split image into R, G, B channels
    channels = [image_np[:, :, i] for i in range(3)]

    # Function to apply wavelet denoising to a single channel
    def denoise_channel(channel):
        coeffs = pywt.wavedec2(channel, wavelet, level=level)

        # Thresholding (soft) - apply threshold to detail coefficients only
        threshold = np.median(np.abs(coeffs[-1])) / 0.6745

        # Apply thresholding to each detail coefficient tuple (horizontal, vertical, diagonal)
        new_coeffs = [coeffs[0]]  # Keep the approximation coefficients unchanged
        for detail_coeffs in coeffs[1:]:
            new_detail_coeffs = tuple(pywt.threshold(c, threshold, mode='soft') for c in detail_coeffs)
            new_coeffs.append(new_detail_coeffs)

        # Reconstruct the channel
        return pywt.waverec2(new_coeffs, wavelet)

    # Apply denoising to each channel
    denoised_channels = [denoise_channel(channel) for channel in channels]

    # Recombine the R, G, B channels
    denoised_image = np.stack(denoised_channels, axis=-1)
    return np.clip(denoised_image.astype(np.float32), 0., 255.)


class FilterRemoval(Removal):
    """
        Low-pass filter on image
    """
    def __init__(self, args):
        super().__init__(args)
    
    def remove_watermark(self, original_image):
        # FFT-based low-pass filter
        removals = []
        removals.append(remove_noise_fft_rgb(original_image, cutoff=50))
        removals.append(remove_noise_wavelet_rgb(original_image, level=1))

        # Average the pixel values across the images
        average_image_array = np.mean(np.stack(removals), axis=0)
        average_image_array = average_image_array.astype(np.uint8)

        # Convert the averaged array back to an image
        average_image = Image.fromarray(average_image_array)

        return average_image
