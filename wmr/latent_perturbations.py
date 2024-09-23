import torch as ch
import numpy as np
from wmr.config import ExperimentConfig
import os

from wmr.base import Removal
from wmr.attack_utils import smimifgsm_attack
from torchvision import transforms
from torchvision.models import resnet18

from transformers import CLIPProcessor, CLIPModel, AutoModel

from diffusers import AutoencoderKL, DiffusionPipeline, ConsistencyDecoderVAE


def load_surrogate_model(path, device: str):
    model = resnet18(pretrained=False)
    model.fc = ch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ch.load(path, weights_only=True))
    model.eval()
    model = model.to(device)
    # Compile model
    model = ch.compile(model)
    return model


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


class ClassificationWrapper(VAEModelWrapper):
    def __call__(self, x):
        return self.model(x)


class CLIPWrapper(VAEModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self._mean = [0.48145466, 0.4578275, 0.40821073]
        self._std = [0.26862954, 0.26130258, 0.27577711]
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=self._mean, std=self._std),
            ]
        )

    def __call__(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.get_image_features(**inputs)
        return outputs


class DINOWrapper(VAEModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.Normalize(mean=self._mean, std=self._std),
            ]
        )
    
    def __call__(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states


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
        # Could also consider adding:
        # - CLIP (https://github.com/umd-huang-lab/WAVES/blob/main/adversarial/feature_extractors/clip.py)
        # clip_vit_base_patch32 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        # - Resnet18 (https://github.com/umd-huang-lab/WAVES/blob/main/adversarial/feature_extractors/resnet18.py)
        # AND additionally fool detection models (together)
        # dinov2_base = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)

        # TODO: Do not hardcode this path
        MODELS_PATH_ = "/home/groot/work/erasing-the-invisible/models"

        # Load up watermark detection model(s)
        stable_sig = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_stable_sig.pth"), self.device)
        tree_ring = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_tree_ring.pth"), self.device)
        stegastamp = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_stegastamp.pth"), self.device)

        self.model = {
            "CompVis/stable-diffusion-v1-4": (VAEModelWrapper(stable_diffusion_v1_4), "embed"),
            "openai/consistency-decoder": (VAEModelWrapper(consistency_decoder), "embed"),
            # "openai/clip-vit-base-patch32": (CLIPWrapper(clip_vit_base_patch32), "embed"),
            # "facebook/dinov2-base": (DINOWrapper(dinov2_base), "embed"),
            # "stabilityai/sdxl-vae": (VAEModelWrapper(sdxl_vae), "embed")
            "stable_sig": (ClassificationWrapper(stable_sig), "classify"),
            "tree_ring": (ClassificationWrapper(tree_ring), "classify"),
            "stegastamp": (ClassificationWrapper(stegastamp), "classify")
        }

    def _remove_watermark(self, original_image):
        image_tensor = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        with ch.no_grad():
            target_embeddings = {}
            for k, (v, v_type) in self.model.items():
                if v_type == "embed":
                    target_embeddings[k] = v(image_tensor).detach()

        perturbed_image_tensor = smimifgsm_attack(self.model,
                                                  image_tensor,
                                                  eps=16/255,
                                                  n_iters=100,
                                                  num_transformations=20,
                                                  proportional_step_size=False,
                                                  step_size_alpha=1,
                                                  target=target_embeddings,
                                                  device=self.device)
        
        # Convert to numpy and standard format to be later used by PIL
        perturbed_image = perturbed_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        return perturbed_image
