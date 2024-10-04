"""
    Code borrowed from warm-up-kit from challenge organizers.
    https://github.com/erasinginvisible/warm-up-kit/
"""

import torch
import torch.nn as nn
import json
import os
import torch.nn.functional as F

import torch
from transformers import CLIPModel


class AestheticScorer(nn.Module):
    def __init__(
        self,
        input_size: int=0,
        use_activation: bool=False,
        dropout: float=0.2,
        config=None,
        hidden_dim: int=1024,
        reduce_dims: bool=False,
        output_activation=None,
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "use_activation": use_activation,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "reduce_dims": reduce_dims,
            "output_activation": output_activation,
        }
        if config != None:
            self.config.update(config)

        layers = [
            nn.Linear(self.config["input_size"], self.config["hidden_dim"]),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                self.config["hidden_dim"],
                round(self.config["hidden_dim"] / (2 if reduce_dims else 1)),
            ),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                round(self.config["hidden_dim"] / (2 if reduce_dims else 1)),
                round(self.config["hidden_dim"] / (4 if reduce_dims else 1)),
            ),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                round(self.config["hidden_dim"] / (4 if reduce_dims else 1)),
                round(self.config["hidden_dim"] / (8 if reduce_dims else 1)),
            ),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Linear(round(self.config["hidden_dim"] / (8 if reduce_dims else 1)), 1),
        ]
        if self.config["output_activation"] == "sigmoid":
            layers.append(nn.Sigmoid())
        layers = [x for x in layers if x is not None]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.config["output_activation"] == "sigmoid":
            upper, lower = 10, 1
            scale = upper - lower
            return (self.layers(x) * scale) + lower
        else:
            return self.layers(x)

    def save(self, save_name):
        split_name = os.path.splitext(save_name)
        with open(f"{split_name[0]}.config", "w") as outfile:
            outfile.write(json.dumps(self.config, indent=4))

        for i in range(
            6
        ):  # saving sometiles fails, so retry 5 times, might be windows issue
            try:
                torch.save(self.state_dict(), save_name)
                break
            except RuntimeError as e:
                # check if error contains string "File"
                if "cannot be opened" in str(e) and i < 5:
                    print("Model save failed, retrying...")
                else:
                    raise e


def preprocess(embeddings):
    return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)


def load_model(weight_name, device: str="cuda"):
    # TODO: Do not hardcode this path
    MODELS_PATH_ = "/home/groot/work/erasing-the-invisible/models/aesthetics"
    weight_path = os.path.join(MODELS_PATH_, f"{weight_name}.pth")
    config_path = os.path.join(MODELS_PATH_, f"{weight_name}.config")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    model = AestheticScorer(config=config)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model


def clip_preprocess(image: torch.Tensor) -> torch.Tensor:
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image.device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image.device)
    
    crop_size = 224

    # Define the transformations
    # Resize the image
    resized_image = F.interpolate(image, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
    # image = kornia_geo.resize(image, size=(crop_size, crop_size)) 
    # Center crop the image (Redundant when same size)
    # image = kornia.center_crop(image, output_size=(crop_size, crop_size))
    # Normalize the image
    # resized_image = (resized_image - image_mean) / image_std
    return resized_image


def load_aesthetics_and_artifacts_models(device: str = "cuda"):
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    vision_model = model.vision_model
    vision_model.to(device)
    del model
    #clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    rating_model = load_model("aesthetics_scorer_rating_openclip_vit_h_14").to(device)
    artifacts_model = load_model("aesthetics_scorer_artifacts_openclip_vit_h_14").to(
        device
    )
    return vision_model, rating_model, artifacts_model
