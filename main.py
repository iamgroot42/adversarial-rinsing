"""
    Read given data, apply watermark-removal function, and generate images for submission.
"""
from simple_parsing import ArgumentParser
import os
from PIL import Image
from datetime import date
from tqdm import tqdm

from wmr.utils import get_method


def main(args):
    # Read images
    images = read_images(args.track)

    # Initialize watermark-removal function
    method = get_method(args.method)(args)

    # Apply watermark-removal function
    watermark_removed_images = []
    for image in tqdm(images, desc="Removing watermarks"):
        watermark_removed_images.append(method.remove_watermark(image))

    # Save images for submission, and add today's date to name
    submission_folder = os.path.join("submissions", args.track, f"{args.method}_{args.submission}_{date.today()}")
    os.makedirs(submission_folder, exist_ok=True)

    for i, image in enumerate(watermark_removed_images):
        image.save(os.path.join(submission_folder, f"{i}.png"))
    
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
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--track", type=str, choices=["black", "beige"], default="black", help="Track to target (black/beige)")
    arg_parser.add_argument("--submission", type=str, default="submission", help="Name for submission attempt")
    arg_parser.add_argument("--method", type=str, default="submission", help="Method to use for watermark removal")
    args = arg_parser.parse_args()

    main(args)
