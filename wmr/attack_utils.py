import numpy as np
import torch as ch
import gc
from torch.autograd import Variable as V
import torch.nn.functional as F
import pywt

def introduce_discontinuity(x: np.ndarray, block_size: int):
    """
        Given an array that contains multiple transforms of the same image,
        construct a new image that samples (block_size, block_size) blocks
        randomly from the input array. Also do this across channels.
        x wil be a 4D array of size (num_transforms, H, W, C).
        Hypothesis: introducing discontinuity in the image will make the watermark less likely to persist.
    """
    num_transforms, H, W, C = x.shape
    assert num_transforms > 1, "Need at least 2 transforms"
    assert C == 3, "Image should be RGB"

    new_x = np.zeros(x.shape[1:], dtype=x.dtype)

    for i in range(H // block_size):
        for j in range(W // block_size):
            for k in range(3):
                random_source = np.random.randint(0, num_transforms)
                new_x[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, k] = x[random_source, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, k]

    return new_x


def average_images(x: np.ndarray):
    """
        Given an array that contains multiple transforms of the same image,
        average the pixel values across the images.
        x wil be a 4D array of size (num_transforms, H, W, C).
    """
    # Average the pixel values across the images
    return np.mean(np.stack(x), axis=0)


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


def transformation_function(x, resize_to: int = 270):
    img_size = x.shape[-1]
    img_resize = resize_to
    rnd = ch.randint(low=img_resize, high=img_size, size=(1,), dtype=ch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_size - rnd
    w_rem = img_size - rnd
    pad_top = ch.randint(low=0, high=h_rem.item(), size=(1,), dtype=ch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = ch.randint(low=0, high=w_rem.item(), size=(1,), dtype=ch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def smimifgsm_attack(aux_models: dict,
                     x_orig: ch.Tensor,
                     eps: float = 4/255,
                     n_iters: int = 100,
                     step_size_alpha: float = 2.5,
                     target=None,
                     target_is_embedding: bool = False,
                     device: str = "cuda"):
    """
        Adapted from SMIMIFGSM implementation in https://github.com/iamgroot42/blackboxsok
        Supports optimization to fool some target classifier, or simply maximize distance from some embedding
    """
    move_direction = 1
    if target_is_embedding:
        # We want to maximize L2 norm distance from given target embeddin
        criterion = ch.nn.MSELoss()
    else:
        # We want to minimize the target, unless targeted is True
        criterion = ch.nn.BCEWithLogitsLoss()
        if target is not None:
            move_direction = -1

    if not isinstance(aux_models, dict):
        raise ValueError("Expected a dictionary of auxiliary models, even if single model provided")
    # temporarily set these values for testing based on their original tf implementation
    x_min_val, x_max_val = 0, 1.0

    n_model_ensemble = len(aux_models)
    alpha = step_size_alpha * eps / n_iters
    decay = 1.0
    momentum = 0
    num_transformations = 12
    lamda = 1 / num_transformations
    resize_to = int(0.9 * 512)

    # initializes the advesarial example
    # x.requires_grad = True
    adv = x_orig.clone()
    adv = adv.to(device)
    adv.requires_grad = True
    x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
    x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)

    for model_name in aux_models:
        model = aux_models[model_name]
        model.set_eval()  # Make sure model is in eval model
        model.zero_grad()  # Make sure no leftover gradients

    i = 0
    while i < n_iters:

        """
        with ch.no_grad():
            losses = []
            for model_name in aux_models:
                model = aux_models[model_name]
                output = model(adv)
                losses.append(criterion(output, target[model_name]).item())
            print("Loss:", np.mean(losses), "Losses:", losses)
        """

        Gradients = []
        if adv.grad is not None:
            adv.grad.zero_()
        if i == 0:
            adv = clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad=True)
        grad = 0

        for t in range(num_transformations):
            adv = adv
            output = 0
            loss = 0
            for model_name in aux_models:
                model = aux_models[model_name]
                if target_is_embedding:
                    output = model(transformation_function(adv, resize_to=resize_to))
                    output_clone = output.clone()
                    # Use target embedding specific to this model
                    loss += criterion(output, target[model_name]) / n_model_ensemble
                else:
                    output += model(transformation_function(adv, resize_to=resize_to)) / n_model_ensemble

            if not target_is_embedding:
                output_clone = output.clone()
                loss = criterion(output_clone, target)

            loss.backward()
            Gradients.append(adv.grad.data)

        for gradient in Gradients:
            grad += lamda * gradient

        grad = momentum * decay + grad / ch.mean(ch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = grad

        adv = adv + (alpha * ch.sign(grad) * move_direction)
        adv = clip_by_tensor(adv, x_min, x_max)
        adv = V(adv, requires_grad=True)

        # outputs the transferability
        model.set_eval()  # Make sure model is in eval model
        model.zero_grad()  # Make sure no leftover gradients

        del output, output_clone
        ch.cuda.empty_cache()
        del loss
        gc.collect()  # Explicitly call the garbage collector

        i += 1

    return adv.detach()
