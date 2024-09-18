"""
    Look at individual images manually and see if there are any patterns,
    based on image transforms.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import pywt
import pywt.data
from PIL import Image

# Load image
original = pywt.data.camera()

INDEX = 0
images_path = os.path.join("data", "Neurips24_ETI_BlackBox", f"{INDEX}.png")

image = Image.open(images_path)
original = np.array(image)
# Get only R channel
original = original[:, :, 2]

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.savefig('inspect.png')