import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def histogram(image):
    freq = np.zeros(256)
    for pxl in image.ravel():
        freq[pxl] += 1 
    
    return freq 

def histogram_equalisation(image):
    M, N = image.shape
    hist = histogram(image)
    cdf = np.cumsum(hist)
    smallest = np.min(cdf[cdf > 0])

    new_pixel = np.zeros(256)
    for i in range(256):
        new_pixel[i] = ((cdf[i] - smallest) / (M*N - smallest)) * 255

    new_image = new_pixel[image]

    return new_image.astype(np.uint8)


image = Image.open('./images/highway.jpg').convert('L')
image = np.array(image)
eq = histogram_equalisation(np.array(image))

fig, axes = plt.subplots(2, 2, figsize=(12,12))
axes = axes.ravel()

axes[0].imshow(image, cmap="gray")
axes[0].axis('off')
axes[0].set_title("Original image")

axes[1].bar(np.arange(256), histogram(image))
axes[1].set_title("Histogram of original image")

axes[2].imshow(eq, cmap="gray")
axes[2].axis('off')
axes[2].set_title("Equalised image")

axes[3].bar(np.arange(256), histogram(eq))
axes[3].set_title("Histogram of equalised image")

plt.show()