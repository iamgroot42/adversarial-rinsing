import torch as ch
import numpy as np
from wmr.config import ExperimentConfig

from wmr.base import Removal
from wmr.attack_utils import smimifgsm_attack

from transformers import CLIPFeatureExtractor
from diffusers import AutoencoderKL, DiffusionPipeline, ConsistencyDecoderVAE
import torch


class VAEModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x_ = x # Will be normalized internally
        latent_mean = self.model.encode(x_).latent_dist.mean
        return latent_mean

    def set_eval(self):
        self.model.eval()
    
    def zero_grad(self):
        self.model.zero_grad()


class VAERemoval(Removal):
    """
        Interfere with "latent" noise for a generative model to
        disrupt latent space as much as possible with a constraint on the input norm.
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # CompVis/stable_diffusion_v1_4
        stable_diffusion_v1_4 = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        # openai/consistency-decoder
        consistency_decoder = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder").to(self.device)
        # sdxl-vae
        # sdxl_vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(self.device)

        self.model = {
            "CompVis/stable-diffusion-v1-4": VAEModelWrapper(stable_diffusion_v1_4),
            "openai/consistency-decoder": VAEModelWrapper(consistency_decoder),
            # "stabilityai/sdxl-vae": VAEModelWrapper(sdxl_vae)
        }
    
    def _remove_watermark(self, original_image):
        image_tensor = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        with ch.no_grad():
            target_embeddings = {}
            for k, v in self.model.items():
                target_embeddings[k] = v(image_tensor).detach()

        perturbed_image_tensor = smimifgsm_attack(self.model, image_tensor,
                                                  eps=8/255, n_iters=500,
                                                  num_transformations=50,
                                                  step_size_alpha=5,
                                                  target=target_embeddings,
                                                  target_is_embedding=True,
                                                  device=self.device)
        
        # Convert to numpy and standard format to be later used by PIL
        perturbed_image = perturbed_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        return perturbed_image
