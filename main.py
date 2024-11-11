"""
    Read given data, apply watermark-removal function, and generate images for submission.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from PIL import Image
from datetime import date
from tqdm import tqdm

from wmr.utils import get_method
from wmr.config import ExperimentConfig

import torch
torch.set_float32_matmul_precision('high')


def main(config: ExperimentConfig):
    # Read images
    images = read_images(config.track)

    # Initialize watermark-removal function
    methods = [get_method(method)(config) for method in config.methods]

    methods_combined_name = "+".join(config.methods)

    # Save images for submission, and add today's date to name
    subname = f"{config.aggregation}_{methods_combined_name}_{config.submission}_{date.today()}"
    if config.track != "test":
        submission_folder = os.path.join("submissions", config.track, subname)
        os.makedirs(submission_folder, exist_ok=True)

    for i, image in tqdm(enumerate(images), desc="Removing watermarks", total=len(images)):
        # If not test-track and image exists, skip
        if config.track != "test" and os.path.exists(os.path.join(submission_folder, f"{i}.png")):
            continue

        watermarked_image = image
        # Apply watermark-removal function iteratively
        for method in methods:
            watermarked_image = method.remove_watermark(watermarked_image)

        if config.track != "test":
            watermarked_image.save(os.path.join(submission_folder, f"{i}.png"))
        else:
            watermarked_image.save(os.path.join("submissions", config.track, f"{subname}.png"))

    if config.skip_zip or config.track == "test":
        return

    # Zip this folder (images inside it, not the entire folder)
    os.system(f"zip -jr {submission_folder}.zip {submission_folder}")

    # Delete the folder (we just need the .zip)
    os.system(f"rm -r {submission_folder}")


def read_images(track: str):
    name_mapping = {
        "black": "BlackBox",
        "beige": "BeigeBox",
        "test": "Test"
    }
    images_path = os.path.join("data", f"Neurips24_ETI_{name_mapping[track]}")

    # Read all .png images in this folder, in order
    # Count number of files that end with .png
    num_images = len([name for name in os.listdir(images_path) if name.endswith(".png")])
    if track != "test":
        if num_images != 300:
            raise ValueError(f"Expected 300 images for black/beige track, found {num_images}.")
    else:
        if num_images != 1:
            raise ValueError(f"Expected 1 image for test track, found {num_images}.")

    images = []
    for i in range(num_images):
        image_path = os.path.join(images_path, f"{i}.png")
        image = Image.open(image_path)
        images.append(image)
    return images


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", help="Specify config file", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)

    main(config)
