import torch as ch
import numpy as np
from PIL import Image

from wmr.base import Removal
from bbeval.attacker.transfer_methods import SMIMIFGSM
from bbeval.config import TransferredAttackConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper


class Adversarial(Removal):
    """
        Add averaged iterations for SMIMIFGSM across multiple trials.
        Target low perturbation budget. Doesn't matter what the "class" corresponds to
        for the detection model. We just want to completely flip a dummy model's behavior.
    """
    def __init__(self, args):
        super().__init__(args)
        # Initialize model that we will use for the attack
        # Initialize attacker
        self.attacker = SMIMIFGSM(
            model=self.model, # TODO: wrap in GenericModelWrapper,
            aux_models=aux_models,
            config=config, # AttackerConfig
            experiment_config=experiment_config # ExperimentConfig
        )
    
    def remove_watermark(self, original_image):
        # Prepare version of image that can be used as pytorch vector
        original_image_pt = ch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        
        num_trials = 10
        adv_images = []
        for _ in range(num_trials):
            adv_image, _ = self.attacker.attack(original_image_pt)
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
