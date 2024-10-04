import numpy as np
import torch as ch
import gc
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision.transforms import v2
from diff_jpeg import DiffJPEGCoding
import pywt
import kornia
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from wmr.aesthetics import load_aesthetics_and_artifacts_models, clip_preprocess
from wmr.differentiable_histogram import differentiable_histogram

import lpips


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
            random_source = np.random.randint(0, num_transforms)
            for k in range(3):
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


def transformation_function(image, resize_to: int = 270, mixup_data=None):

    def random_crop(x):
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

    def horizontal_flip(x):
        return v2.functional.hflip(x)
        
    def random_brightness(x):
        brightness_factor = np.random.uniform(0.05, 0.3)
        return kornia.enhance.adjust_brightness(x, brightness_factor)

    def random_contrast(x):
        contrast_factor = np.random.uniform(0.05, 0.3)
        return kornia.enhance.adjust_contrast(x, contrast_factor)
    
    def random_saturation(x):
        saturation_factor = np.random.uniform(-0.3, 0.3)
        return kornia.enhance.adjust_saturation(x, saturation_factor)
    
    def random_hue(x):
        hue_factor = np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
        return kornia.enhance.adjust_hue(x, hue_factor)
    
    def gaussian_blur(x):
        kernel_size = np.random.randint(6, 25)
        # Make sure kernel-size is odd
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
        return blur(x)

    def gaussian_noise(x):
        std = np.random.uniform(0.05, 0.2)
        return x + ch.randn_like(x) * std

    def jpeg_compression(x):
        quality = np.random.randint(70, 90)
        jpeg_module = DiffJPEGCoding()
        quality = ch.tensor([quality]).to(x.device)
        jpeg = jpeg_module(x, quality)
        return jpeg

    def fft_noise(x):
        std = np.random.uniform(0.05, 0.2)
        # Differentiable FFT noise
        fft_img = ch.fft.fft2(x)
        # Add noise
        fft_noisy_img = fft_img + ch.randn_like(x) * std
        # Convert noisy image back to spatial domain
        return ch.fft.ifft2(fft_noisy_img).real

    def rotation(x):
        # Random rotation in [9, 45] clockwise
        angle = np.random.randint(5, 45)
        return v2.functional.rotate(x, int(angle))

    def motion_blur(x):
        angle = np.random.randint(5, 175)
        direction = np.random.choice([-1, 1])
        return kornia.filters.motion_blur(x, kernel_size=15,
                                          direction=direction,
                                          angle=angle, border_type='constant')

    def mixup(x):
        # Take a random image from the mixup data
        random_index = np.random.randint(0, len(mixup_data))
        #0 Randomly select a mixup factor in [0.1, 0.3]
        mixup_factor = np.random.uniform(0.1, 0.3)
        mixup_image = mixup_data[random_index].unsqueeze(0).to(x.device)
        mixed = x * (1 - mixup_factor) + mixup_image * mixup_factor
        return mixed
        sharpness_factor = np.random.uniform(1 - 0.25, 1 + 0.25)
        return v2.functional.adjust_sharpness(x, sharpness_factor)

    transformation_functions = [
        random_crop,
        gaussian_blur,
        gaussian_noise,
        jpeg_compression,
        fft_noise,
        rotation,
        motion_blur,
        random_brightness,
        random_contrast,
        random_saturation,
        random_hue,
        horizontal_flip,
        mixup
    ]
    # Randomly pick one of the transformation functions
    random_transform = transformation_functions[np.random.randint(0, len(transformation_functions))]
    return random_transform(image), random_transform


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



class MSEandCosine(ch.nn.Module):
    def __init__(self, alpha: float=0.5):
        super().__init__()
        self.alpha = alpha  # weight for combining MSE and cosine distance
        self.mse = ch.nn.MSELoss()
        self.csn = ch.nn.CosineSimilarity()

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        # Flatten to compute cosine similarity
        csn_loss = 1 - self.csn(output.view(output.size(0), -1), target.view(target.size(0), -1))

        # Combined Loss
        loss = (1 - self.alpha) * mse_loss + self.alpha * csn_loss
        return loss


class NormalizedImageQuality(ch.nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        # self.fid = torch_metrics.FrechetInceptionDistance(device)
        
        self.lpips = lpips.LPIPS(net='alex')
        self.lpips.to(self.device)
    
        self.aesthetic_models = load_aesthetics_and_artifacts_models(device)

        self.target_aesthetics = None
        self.target_artifacts = None
    
    def _nmi(self, x, y, bins: int = 100):
        # Compute Normalized Mutual Information (NMI) between two images
        # Use differentiable histogram form kornia
        #_, hist_x = kornia.ehnance.image_histogram2d(x, n_bins=bins, min=0., max=1., return_pdf=True)
        #_, hist_y = kornia.ehnance.image_histogram2d(y, n_bins=bins, min=0., max=1., return_pdf=True)

        def entropy(p):
            p = p + 1e-10  # To avoid log(0)
            return -ch.sum(p * ch.log(p))

        hists = differentiable_histogram(ch.cat([x, y], dim=0), bins=bins, min=0., max=1.)
        H0 = entropy(np.sum(hist, axis=0))
        H1 = entropy(np.sum(hist, axis=1))
        H01 = entropy(np.reshape(hist, -1))

        # return (H0 + H1) / H01

    def _compute_aesthetics_and_artifacts_scores(self, image):
        vision_model, rating_model, artifacts_model = self.aesthetic_models

        def preprocess(x):
            return x / x.norm(p=2, dim=-1, keepdim=True)

        inputs = clip_preprocess(image)

        vision_output = vision_model(pixel_values=inputs)
        pooled_output = vision_output.pooler_output

        embedding = preprocess(pooled_output)

        rating = rating_model(embedding)
        artifact = artifacts_model(embedding)

        return rating.mean(), artifact.mean()

    def _psnr(self, x, y):
        # MSE in pytorch
        mse = F.mse_loss(x, y)
        # For too-low MSE values, PSNR will be too high
        # To avoid division by zero, return a very low value
        if mse < 1e-10:
            return 100
        return - 10 * ch.log10(mse)

    def _ssim(self, x, y):
        # Gaussian kernel
        ssim_val = ssim(x, y, data_range=1.)
        return ssim_val.mean()

    def _lpips(self, x, y):
        return self.lpips(x, y, normalize=True).mean()

    def forward(self, output, target):
        """
            Output here is generated image, target is original image
        """
        #fid_score = self.fid.compute(output, target)

        lpips_score = self._lpips(output, target)
        psnr_score = self._psnr(output, target)
        ssim_score = self._ssim(output, target)
        outputs_aesthetics, outputs_artifacts = self._compute_aesthetics_and_artifacts_scores(output)

        if self.target_aesthetics is None:
            self.target_aesthetics, self.target_artifacts = self._compute_aesthetics_and_artifacts_scores(target)

        delta_aesthetics = outputs_aesthetics - self.target_aesthetics
        delta_artifacts = outputs_artifacts - self.target_artifacts

        # Differentiable metrics!
        weighted_scores = [
            # (fid_score, 1.53e-3),  # FID
            # (None, 5.07e-3), # CLIP FID
            (psnr_score, -2.22e-3), # PSNR, works
            (ssim_score, -1.13e-1), # SSIM, works
            # (None, -9.88e-2), # NMI
            (lpips_score, 3.41e-1), # LPIPS, works
            (delta_aesthetics, 4.5e-2), # Delta-Aesthetics, works
            (delta_artifacts, -1.44e-1), # Delta-Artifacts, works
        ]

        # Aggregate weighted scores
        # print([i[0] for i in weighted_scores])
        final_score = sum([score * weight for score, weight in weighted_scores])
        # We want to minimize this score
        return final_score


def smimifgsm_attack(aux_models: dict,
                     x_orig: ch.Tensor,
                     eps: float = 4/255,
                     n_iters: int = 100,
                     step_size_alpha: float = 2.5,
                     num_transformations: int = 12,
                     proportional_step_size: bool = True,
                     target=None,
                     mixup_data=None,
                     image_quality_metric: ch.nn.Module = None,
                     image_quality_multiplier: float = -1e6,
                     device: str = "cuda"):
    """
        Adapted from SMIMIFGSM implementation in https://github.com/iamgroot42/blackboxsok
        Supports optimization to fool some target classifier, or simply maximize distance from some embedding
    """
    mse_criterion = MSEandCosine(alpha=0) # Ignore cosine similarity for the time being
    classification_criterion = ch.nn.CrossEntropyLoss()
    
    class_one = ch.tensor([1]).to(device)
    classification_ease_factor = 0.05

    if not isinstance(aux_models, dict):
        raise ValueError("Expected a dictionary of auxiliary models, even if single model provided")
    # temporarily set these values for testing based on their original tf implementation
    x_min_val, x_max_val = 0, 1.0

    n_model_ensemble = len(aux_models)
    n_model_embed = sum([1 for model_name in aux_models if aux_models[model_name][1] == "embed"])
    n_model_clasify = n_model_ensemble - n_model_embed

    if proportional_step_size:
        alpha = step_size_alpha * eps / n_iters
    else:
        alpha = step_size_alpha
    decay = 1.0
    momentum = 0
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
        model, _ = aux_models[model_name]
        model.zero_grad()  # Make sure no leftover gradients

    i = 0
    while i < n_iters:

        # """
        with ch.no_grad():
            losses = []
            for model_name in aux_models:
                model, model_type = aux_models[model_name]
                output = model(adv)
                if model_type == "embed":
                    loss_ = mse_criterion(output, target[model_name])
                else:
                    loss_ = classification_criterion(output, class_one) * classification_ease_factor
                losses.append(loss_.item())

            if image_quality_metric is not None:
                qual_loss = image_quality_metric(adv, x_orig).item()
            print(f"Step {i+1}/{n_iters} | Quality Loss:", qual_loss, "| Loss:", np.mean(losses), "| Losses:", losses)
        # """

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
                model, model_type = aux_models[model_name]

                if model_type == "embed":
                    transformed_image, tf_info = transformation_function(adv, resize_to=resize_to, mixup_data=mixup_data)
                    embed_output = model(transformed_image)
                    # Use target embedding specific to this model
                    loss += mse_criterion(embed_output.clone(), target[model_name]) / n_model_embed
                else:
                    transformed_image, tf_info = transformation_function(adv, resize_to=resize_to, mixup_data=mixup_data)
                    output += model(transformed_image) / n_model_clasify

            if model_type != "embed":
                output_clone = output.clone()
                # CE is very easy to optimize, so adjust loss ti be smaller
                loss += classification_criterion(output_clone, class_one) * classification_ease_factor
            
            # Add image quality metric if specified
            if image_quality_metric is not None:
                # TODO: Should consider much-higher weight for this loss (relative)
                loss += image_quality_metric(adv, x_orig) * image_quality_multiplier

            loss.backward()
            Gradients.append(adv.grad.data)

        for gradient in Gradients:
            grad += lamda * gradient

        grad = momentum * decay + grad / ch.mean(ch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = grad

        adv = adv + alpha * ch.sign(grad)
        adv = clip_by_tensor(adv, x_min, x_max)
        adv = V(adv, requires_grad=True)

        del output
        if n_model_clasify > 0:
            del output_clone
        ch.cuda.empty_cache()
        del loss
        gc.collect()  # Explicitly call the garbage collector

        i += 1

    return adv.detach()
