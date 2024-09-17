# Similar logic to adversarial- add perturbations to images
# until exiting watermark detection models are fooled.
# alternatively, could target watermark-extraction methods directly
# or perhaps joint optimization for both?

import torch as ch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18

from wmr.base import Removal
from wmr.utils import get_models_path, PyTorchModelWrapper

from bbeval.attacker.transfer_methods import SMIMIFGSM
from bbeval.config import TransferredAttackConfig, AttackerConfig, ExperimentConfig, ModelConfig


def prepare_detection_model(model_path: str):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ch.load(model_path))
    model.eval()
    model = ch.compile(model)

    wrapped_model = PyTorchModelWrapper(model)
    return wrapped_model


class BypassDetection(Removal):
    """
        Similar logic to adversarial- add perturbations to images until exiting watermark detection models are fooled.
        Alternatively, could target watermark-extraction methods directly, or perhaps joint optimization for both?
    """
    def __init__(self, args):
        super().__init__(args)
        # We want real/wm to say "real", and unwm/wm to say "unwm"
        # For all three available detection methods, so targeting 6 models all in all

        # TODO: Add missing entries required for config
        attacker_config = AttackerConfig(
            name="SMIMIFGSM_transfer",
            eps=4.0,
            targeted=True,
            norm_type=np.inf,
            track_global_metrics=False,
            track_local_metrics=False,
            time_based_attack=False
        )
        config = ExperimentConfig(
            experiment_name="bypass_detection",

        )
        
        # All models in get_models_path that start with "adv_cls_" are detection models
        aux_models = {}
        for model_name in os.listdir(get_models_path()):
            if model_name.startswith("adv_cls_"):
                desired_path = os.path.join(get_models_path(), model_name)
                model = prepare_detection_model(desired_path)
                # Remove "pth" from end
                aux_models[model_name[:-4]] = model

        # Desired target class is 0

        # Initialize attacker
        self.attacker = SMIMIFGSM(
            model=self.model, # TODO: wrap in GenericModelWrapper,
            aux_models=aux_models,
            config=attacker_config, # AttackerConfig
            experiment_config=ExperimentConfig # ExperimentConfig
        )
    
    def remove_watermark(self, original_image):
        # Prepare version of image that can be used as pytorch vector
        original_image_pt = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        
        num_trials = 10
        adv_images = []
        for _ in range(num_trials):
            adv_image, _ = self.attacker.attack(original_image_pt, y_target=[0])
            adv_images.append(adv_image)

        # Average the adversarial images
        average_image_array = np.zeros_like(np.array(original_image))
        for adv_image in adv_images:
            adv_image_array = adv_image.squeeze().permute(1, 2, 0).cpu().numpy()
            average_image_array += adv_image_array

        average_image_array *= 255.0
        average_image_array /= num_trials
        average_image_array = np.clip(average_image_array, 0, 255).astype(np.uint8)

        # Convert the averaged array back to an image
        average_image = Image.fromarray(average_image_array)

        return average_image
