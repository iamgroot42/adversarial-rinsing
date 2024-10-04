import torch as ch
import numpy as np
from wmr.config import ExperimentConfig
import os
from PIL import Image
from tqdm import tqdm

from wmr.base import Removal
from wmr.attack_utils import smimifgsm_attack, clip_by_tensor, NormalizedImageQuality
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from transformers import CLIPProcessor, CLIPModel, AutoModel

from diffusers import AutoencoderKL, DiffusionPipeline, ConsistencyDecoderVAE

from wmr.diffusion_utils import ReSDPipeline, DiffWMAttacker


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
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        return decoded_image


class DiffusionModelWrapper(VAEModelWrapper):
    def __init__(self, model):
        super().__init__(model.vae)

    # def __call__(self, x):
    #     x_ = x * 2 - 1
    #     latent_mean = self.model.vae.encode(x_).latent_dist.mean
    #     return latent_mean

    # def zero_grad(self):
    #     self.model.vae.zero_grad()


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
        self.attack_config = self.config.attack_config
        # CompVis/stable_diffusion_v1_4
        # stable_diffusion_v1_4 = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        stable_diffusion_v2_1 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(self.device)
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
        # MODELS_PATH_ = "/home/groot/work/erasing-the-invisible/models/wmd"
        # Load up watermark detection model(s)
        # stable_sig = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_stable_sig.pth"), self.device)
        # tree_ring = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_tree_ring.pth"), self.device)
        # stegastamp = load_surrogate_model(os.path.join(MODELS_PATH_, f"adv_cls_unwm_wm_stegastamp.pth"), self.device)

        self.model = {
            # "CompVis/stable-diffusion-v1-4": (VAEModelWrapper(stable_diffusion_v1_4), "embed"),
            "stabilityai/stable-diffusion-2-1": (DiffusionModelWrapper(stable_diffusion_v2_1), "embed"),
            "openai/consistency-decoder": (VAEModelWrapper(consistency_decoder), "embed"),
            # "stabilityai/stable-diffusion-3-medium-diffusers": (VAEModelWrapper(stable_difffusion_3), "embed"),
            # "resnet18": (ResnetWrapper(resnet_18), "embed"),
            # "openai/clip-vit-base-patch32": (CLIPWrapper(clip_vit_base_patch32), "embed"),
            # "facebook/dinov2-base": (DINOWrapper(dinov2_base), "embed"),
            # "stabilityai/sdxl-vae": (VAEModelWrapper(sdxl_vae), "embed")
            # "stable_sig": (ClassificationWrapper(stable_sig), "classify"),
            # "tree_ring": (ClassificationWrapper(tree_ring), "classify"),
            # "stegastamp": (ClassificationWrapper(stegastamp), "classify")
        }

        self.mixup_images = []
        mixup_data_path = "/home/groot/work/erasing-the-invisible/augmentation_data"
        # Open and read all .jpg images in mixup_data_path
        for filename in os.listdir(mixup_data_path):
            if filename.endswith(".jpg"):
                image = ch.tensor(np.array(Image.open(os.path.join(mixup_data_path, filename)))).permute(2, 0, 1).unsqueeze(0) / 255.0
                self.mixup_images.append(image)
        self.mixup_images = ch.cat(self.mixup_images, dim=0)

        self.regeneration_pipe = ReSDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
        )
        self.regeneration_pipe.to(self.device)

        # self.image_quality_metric = None
        self.image_quality_metric = NormalizedImageQuality(self.device)

    def _rinse_cycle(self, image_tensor,
                     eps:float,
                     n_iters: int,
                     decoding_model: str):
        with ch.no_grad():
            target_embeddings = {}
            for k, (v, v_type) in self.model.items():
                if v_type == "embed":
                    target_embeddings[k] = v(image_tensor).detach()

        # Input: [0, 1]
        # Output: [0, 1]
        perturbed_image_tensor = smimifgsm_attack(self.model,
                                                  image_tensor,
                                                  eps=eps,
                                                  n_iters=n_iters,
                                                  num_transformations=self.attack_config.num_transformations,
                                                #   num_transformations=10,
                                                  proportional_step_size=False,
                                                  step_size_alpha=self.attack_config.step_size_alpha,
                                                #   step_size_alpha=5e-4,
                                                  target=target_embeddings,
                                                  mixup_data=self.mixup_images,
                                                  image_quality_metric=self.image_quality_metric,
                                                  device=self.device)
    
        if decoding_model == "waves":
            # Regeneration, as done exactly in WAVES
            # Sample a "strength" in [50, 200] range (int)
            #strength = np.random.randint(10, 20)
            strength = 10
            rinse_attacker = DiffWMAttacker(self.regeneration_pipe, noise_step=strength)
            perturbed_image_tensor = rinse_attacker.attack(perturbed_image_tensor)
        else:
            # Decode with target model
            # Input: [0, 1] (converts to [-1, 1] internally).
            # Output: [0, 1] range (converts from [-1, 1] internally).
            focus_model = self.model[decoding_model][0]
            perturbed_image_emb = focus_model(perturbed_image_tensor)
            perturbed_image_tensor = focus_model.decode(perturbed_image_emb)

        return perturbed_image_tensor

    def _remove_watermark(self, original_image):
        image_tensor = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        perturbed_image_tensor = image_tensor.to(self.device)

        # Rinse cyclex3 [1, 1, 1]
        # perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=25) # 25
        # perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=15) #15
        # perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=10) # 10

        # perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=2/255, n_iters=25) # 25
        # perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=15) #15
        # perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor, eps=1/255, n_iters=10) # 10

        eps_list = self.attack_config.eps_list
        n_iters_list = self.attack_config.n_iters
        decoding_model_list = self.attack_config.decoding_models

        for (eps, n_iters, decoding_model) in zip(eps_list, n_iters_list, decoding_model_list):
            perturbed_image_tensor = self._rinse_cycle(perturbed_image_tensor,
                                                       eps=eval(eps),
                                                       n_iters=int(n_iters),
                                                       decoding_model=decoding_model)

        """
        # Clip to be within 8/255 eps of original image
        capping_eps = 8/255
        x_min = clip_by_tensor(image_tensor - capping_eps, 0, 1)
        x_max = clip_by_tensor(image_tensor + capping_eps, 0, 1)
        perturbed_image_tensor = clip_by_tensor(perturbed_image_tensor, x_min, x_max)
        """

        # Convert to numpy and standard format to be later used by PIL
        perturbed_image = perturbed_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255

        return perturbed_image
