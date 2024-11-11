# Adversarial Rinsing 😈🧼

This repository includes my submission to the [Erasing the Invisible](https://erasinginvisible.github.io/) challenge at NeurIPS, 2024. I developed the same technique for both black-box and beige-box tracks.

## Description

The technique relies on combining "rinsing" (running an image through a generative model multiple times) and adversarial examples to remove watermarks. For a given set of diffusion models, the algorithm works by first denoising the current image, followed by adversarial perturbations generated using [SMI2FGSM](https://arxiv.org/pdf/2203.13479), which is known to be the [current best adversarial attack](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10516640) under the transfer setting.

The attack is applied using target embeddings with the loss function being a weighted combination of (MSE distance  + cosine distance) between the original and target model's embeddings (to be maximized), and a combination of differentiable approximations of image quality functions, including a combination of PSNR, SSIM, LPIPS, NMI, and Aesthetics/Artifacts scores. The core idea here is to modify the embeddings of the image as much as possible while constraining the change in image pixels (within some $L_\inf$ norm). Since the original watermark function is small enough that the watermarked image is itself acceptable to begin with, trying to push the intermediate embeddings away as much as possible while minimizing image-pixel edits should thus help move away from any latent-space watermarks. At the same time, the rinsing component would take care of any input-space watermarks. Another salient feature of this technique is the controllability of how much the image is perturbed.

For the set of augmentations to be used with SMI2FGSM, I use a combination of multiple differentiable augmentations (approximations used wherever possible) that are commonly used in watermarking algorithms. Thus, the idea here is to generate perturbations that cause a drift in the embedding space when *any* of the said augmentations are applied. I used the following augmentations:

- Random crop
- Gaussian blur
- Gaussian noise
- JPEG compression
- Noise in the FFT domain
- Rotation
- Motion Blur
- Random brightness
- Random contrast
- Random hue
- Horizontal flips
- Mixup (using some clean data downloaded from the Internet)

In my implementation, I found the "waves" and "openai/consistency-decoder" generative models to work best.

## About the code

This repository is set up as a modular package that allows multiple kinds of attacks to be combined and applied sequentially. There is also support for applying all "erasure" algorithms such that they are applied at a randomly-selection of image pixels, with different randomess for each denoising algorithm.
To get a better sense of the configuration files and their layout, please see `wmr/config.py`

## Running the code

1. Make sure you have all the required packages in `requirements.txt`

2. Install this package with `pip install .`. You can install it in edit mode with `-e` if you wish to add your own components to the code.

3. Run any attack with `python main.py --config <CONFIG_PATH>`. Some example configs are:

    - configs/
      - adv_rinse.yml : Adversarial Rinse (described above)
      - compression.yml : Simply compresses images
      - filters.yml: Applies a series of image-filters
      - lowpass.yml: Applies a low-pass filter
