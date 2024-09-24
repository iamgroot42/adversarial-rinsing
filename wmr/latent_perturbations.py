import torch as ch
import numpy as np
from wmr.config import ExperimentConfig
import os

from wmr.base import Removal
from wmr.attack_utils import smimifgsm_attack, clip_by_tensor
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from transformers import CLIPProcessor, CLIPModel, AutoModel

from diffusers import AutoencoderKL, DiffusionPipeline, ConsistencyDecoderVAE


def load_surrogate_model(path, device: str):
    model = resnet18(weights=None)
    model.fc = ch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ch.load(path, weights_only=True))
    model.eval()
    model = model.to(device)
    # Compile model
    # model = ch.compile(model)
    return model


class VAEModelWrapper:
    def __init__(self, model):
        model.eval()
        self.model = ch.compile(model)

    def __call__(self, x):
        # x_ = x # Will be normalized internally
        x_ = x * 2 - 1
        latent_mean = self.model.encode(x_).latent_dist.mean
        return latent_mean
    
    def zero_grad(self):
        self.model.zero_grad()
    
    @ch.no_grad()
    def decode(self, z):
        decoded_image = self.model.decode(z)
        decoded_image = decoded_image.sample
        decoded_image = (decoded_image / 2 + 0.5)#.clamp(0, 1)
        return decoded_image


class DiffusionModelWrapper(VAEModelWrapper):
    def __call__(self, x):
        x_ = x * 2 - 1
        latent_mean = self.model.vae.encode(x_).latent_dist.mean
        return latent_mean

    def zero_grad(self):
        self.model.vae.zero_grad()


class ClassificationWrapper(VAEModelWrapper):
    def __call__(self, x):
        return self.model(x)


class ResnetWrapper(VAEModelWrapper):
    def __init__(self, model):
        fe_model = ch.nn.Sequential(*list(model.children())[:-1])
        super().__init__(fe_model)
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=self._mean, std=self._std),
            ]
        )

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
        # stable_diffusion_v2_1 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(self.device)
        # stable_difffusion_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(self.device)
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
        # resnet_18 = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)

        # TODO: Do not hardcode this path
        MODELS_PATH_ = "/home/groot/work/erasing-the-invisible/models"

        # Load up watermark detection model(s)
        stable_sig = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_stable_sig.pth"), self.device)
        tree_ring = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_tree_ring.pth"), self.device)
        stegastamp = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_stegastamp.pth"), self.device)

        self.model = {
            "CompVis/stable-diffusion-v1-4": (VAEModelWrapper(stable_diffusion_v1_4), "embed"),
            # "stabilityai/stable-diffusion-2-1": (DiffusionModelWrapper(stable_diffusion_v2_1), "embed"),
            "openai/consistency-decoder": (VAEModelWrapper(consistency_decoder), "embed"),
            # "stabilityai/stable-diffusion-3-medium-diffusers": (VAEModelWrapper(stable_difffusion_3), "embed"),
            # "resnet18": (ResnetWrapper(resnet_18), "embed"),
            # "openai/clip-vit-base-patch32": (CLIPWrapper(clip_vit_base_patch32), "embed"),
            # "facebook/dinov2-base": (DINOWrapper(dinov2_base), "embed"),
            # "stabilityai/sdxl-vae": (VAEModelWrapper(sdxl_vae), "embed")
            "stable_sig": (ClassificationWrapper(stable_sig), "classify"),
            "tree_ring": (ClassificationWrapper(tree_ring), "classify"),
            "stegastamp": (ClassificationWrapper(stegastamp), "classify")
        }
    
    def _rinse_cycle(self, image_tensor,
                     eps:float,
                     n_iters: int = 25,
                     decoding_model: str = "openai/consistency-decoder"):
        with ch.no_grad():
            target_embeddings = {}
            for k, (v, v_type) in self.model.items():
                if v_type == "embed":
                    target_embeddings[k] = v(image_tensor).detach()
        
        perturbed_image_tensor = smimifgsm_attack(self.model,
                                                  image_tensor,
                                                  eps=eps,
                                                  n_iters=n_iters,
                                                  num_transformations=20,
                                                  proportional_step_size=False,
                                                  step_size_alpha=1,
                                                  target=target_embeddings,
                                                  device=self.device)
        
        # Decode with target model
        focus_model = self.model[decoding_model][0]
        perturbed_image_emb = focus_model(perturbed_image_tensor)
        perturbed_image_tensor = focus_model.decode(perturbed_image_emb)

        return perturbed_image_tensor

    def _remove_watermark(self, original_image):
        image_tensor = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        # Rinse cyclex3
        perturbed_image_tensor = self._rinse_cycle(image_tensor, eps=1/255, n_iters=25) # 25
        perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=15) #15
        perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=10) # 10

        # Clip to be within 3/255 eps of original image
        capping_eps = 3/255
        x_min = clip_by_tensor(image_tensor - capping_eps, 0, 1)
        x_max = clip_by_tensor(image_tensor + capping_eps, 0, 1)
        perturbed_image_tensor = clip_by_tensor(perturbed_image_tensor, x_min, x_max)

        # Convert to numpy and standard format to be later used by PIL
        perturbed_image = perturbed_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        return perturbed_image
