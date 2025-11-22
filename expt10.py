import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 

def histogram(img):
    img = img.astype(np.uint8)
    freq = np.zeros(256)
    for pxl in img.ravel():
        freq[pxl] += 1
    
    return freq 

def histogram_equalisation(img):
    if len(img.shape) == 3:
        M, N, _ = img.shape
    else:
        M, N = img.shape 

    img = img.astype(np.uint8)

    hist = histogram(img)
    cdf = np.cumsum(hist)
    fmin = np.min(cdf[cdf > 0])

    mapping = np.zeros(256)
    for i in range(256):
        mapping[i] = (cdf[i] - fmin) / (M*N - fmin) * 255 

    res = mapping[img]
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

def rgb_to_hsi(rgb):
    r,g,b = rgb / 255.0
    i = (r+g+b)/3

    if r+g+b != 0:
        s = 1 - 3/(r+g+b) * min(r,g,b)
    else:
        s = 0 
    
    h_num = 1/2 * ((r-g) + (r-b))
    h_den = np.sqrt((r-g)**2 + (r-b)*(g-b)) + 1e-14
    h = np.degrees(np.arccos(h_num / h_den))

    if b > g:
        h = 360.0 - h 
    
    return np.array([h, s, i])

def hsi_to_rgb(hsi):
    h, s, i = hsi 
    h_og = h 
    if 120 <= h < 240:
        h -= 120
    elif 240 <= h < 360:
        h -= 240 

    h = np.radians(h)
    x = i * (1 - s)
    y = i * (1 + s*np.cos(h) / (np.cos(np.pi / 3 - h) + 1e-14))
    z = 3*i - (x+y)

    if 0 <= h_og < 120:
        b,r,g = x,y,z
    elif 120 <= h_og < 240:
        r,g,b = x,y,z
    else:
        g,b,r = x,y,z
    
    res = np.array([r, g, b]) * 255 
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res 
    
def color_histogram_equalisation(img):
    hsi_img = np.apply_along_axis(rgb_to_hsi, axis=2, arr=img)
    hsi_img[:, :, 2] *= 255.0 

    original_histogram = histogram(hsi_img[:, :, 2])
    equalised_intensity = histogram_equalisation(hsi_img[:, :, 2])
    equalised_histogram = histogram(equalised_intensity)

    hsi_img[:, :, 2] = equalised_intensity / 255.0
    
    res = np.apply_along_axis(hsi_to_rgb, axis=2, arr=hsi_img)
    return res, original_histogram, equalised_histogram


image = Image.open('./images/highway.jpg')
op, og_hist, res_hist = color_histogram_equalisation(np.array(image))

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.ravel()

axes[0].axis("off")
axes[0].set_title("Original image")
axes[0].imshow(np.array(image))

axes[1].set_title("Histogram of original image")
axes[1].bar(np.arange(256), og_hist)

axes[2].axis("off")
axes[2].set_title("Equalised image")
axes[2].imshow(op)

axes[3].set_title("Histogram of equalised image")
axes[3].bar(np.arange(256), res_hist)

plt.show()