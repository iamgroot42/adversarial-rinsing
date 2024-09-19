import torch as ch
import numpy as np
from PIL import Image

from wmr.base import Removal
from wmr.attack_utils import smimifgsm_attack

from transformers import CLIPFeatureExtractor
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL
import torch


class VAEModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x_ = 2.0 * x - 1.0
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
    def __init__(self, args):
        super().__init__(args)
        # Load model
        model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        self.model = VAEModelWrapper(model)
    
    def _remove_watermark(self, original_image):
        image_tensor = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        model_dict = {
            "CompVis/stable-diffusion-v1-4": self.model
        }

        with ch.no_grad():
            target_embedding = self.model(image_tensor).detach()

        perturbed_image_tensor = smimifgsm_attack(model_dict, image_tensor,
                                                  eps=1/255, n_iters=50,
                                                  step_size_alpha=5.0,
                                                  target=target_embedding,
                                                  target_is_embedding=True,
                                                  device=self.device)
        
        # Convert to numpy and standard format to be later used by PIL
        perturbed_image = perturbed_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        return perturbed_image


class DiffusionRemoval(Removal):
    """
        Interfere with "latent" noise for a generative model to
        disrupt latent space as much as possible with a constraint on the input norm.
    """
    def __init__(self, args):
        super().__init__(args)
        # Load model
        self.pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(self.device)
    
    def _remove_watermark(self, original_image):
        image_tensor = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = 2.0 * image_tensor - 1.0

        vae = self.pipeline.vae
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.mean

        # Define a timestep (e.g., t = 50)
        t = torch.tensor([50]).to(self.device)

        # Extract noise vector using the diffusion model's UNet at timestep t
        with torch.no_grad():
            predicted_noise = self.pipeline.unet(latent, t).sample

        # TODO: Implement
        return None