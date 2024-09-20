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


def main(config: ExperimentConfig):
    # Read images
    images = read_images(config.track)

    # Initialize watermark-removal function
    methods = [get_method(method)(config) for method in config.methods]

    methods_combined_name = "+".join(config.methods)

    # Save images for submission, and add today's date to name
    submission_folder = os.path.join("submissions", config.track, f"{methods_combined_name}_{config.submission}_{date.today()}")
    os.makedirs(submission_folder, exist_ok=True)

    for i, image in tqdm(enumerate(images), desc="Removing watermarks", total=len(images)):
        watermarked_image = image
        # Apply watermark-removal function iteratively
        for method in methods:
            watermarked_image = method.remove_watermark(watermarked_image)
        watermarked_image.save(os.path.join(submission_folder, f"{i}.png"))

    if config.skip_zip:
        return

    # Zip this folder (images inside it, not the entire folder)
    os.system(f"zip -jr {submission_folder}.zip {submission_folder}")

    # Delete the folder (we just need the .zip)
    os.system(f"rm -r {submission_folder}")


def read_images(track: str):
    trackpathname = "BlackBox" if track == "black" else "BeigeBox"
    images_path = os.path.join("data", f"Neurips24_ETI_{trackpathname}")
    # Read all .png images in this folder, in order
    num_images = 100 if track == "black" else 300
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
