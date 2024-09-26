"""
    Randomly collect images to use in MixUp
"""
import requests
from tqdm import tqdm


def download_image(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download image from", url)


def main(num_images: int):
    for seed in tqdm(range(num_images)):
        url = f"https://picsum.photos/seed/{seed}/512"
        file_path = f"./{seed}.jpg"
        download_image(url, file_path)


if __name__ == "__main__":
    main(300)
