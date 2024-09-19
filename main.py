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

    # Save images for submission, and add today's date to name
    submission_folder = os.path.join("submissions", args.track, f"{args.method}_{args.submission}_{date.today()}")
    os.makedirs(submission_folder, exist_ok=True)

    for i, image in tqdm(enumerate(images), desc="Removing watermarks", total=len(images)):
        # Apply watermark-removal function
        watermarked_image = method.remove_watermark(image)
        watermarked_image.save(os.path.join(submission_folder, f"{i}.png"))

    if args.skip_zip:
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
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--track", type=str, choices=["black", "beige"], default="black", help="Track to target (black/beige)")
    arg_parser.add_argument("--aggregation", type=str, choices=["mean", "random"], default="mean", help="Ways to aggregate multiple augmentations")
    arg_parser.add_argument("--submission", type=str, default="submission", help="Name for submission attempt")
    arg_parser.add_argument("--method", type=str, default="submission", help="Method to use for watermark removal")
    arg_parser.add_argument("--skip_zip", action="store_true", help="Skip zipping the submission folder")
    args = arg_parser.parse_args()

    main(args)
