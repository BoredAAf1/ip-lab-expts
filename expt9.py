import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 


def split_channels(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    return red, green, blue 

def brighten(img, percent):
    img = img.astype(np.int32)
    factor = 1 + (percent/100)
    img = (img * factor)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img 

def contrast(img, alpha):
    img = img.astype(np.int32)
    res = np.zeros_like(img)
    
    for c in [0, 1, 2]:
        channel = img[:, :, c]
        mean = np.mean(channel)

        res[:, :, c] = alpha*(channel - mean) + mean 

    return np.clip(res, 0, 255).astype(np.uint8)

def saturate(img, alpha):
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

    hsi_img = np.apply_along_axis(rgb_to_hsi, axis=2, arr=img)
    hsi_img[:, :, 1] = hsi_img[:, :, 1] * alpha 
    hsi_img[:, :, 1] = np.clip(hsi_img[:, :, 1], 0, 1)
    res = np.apply_along_axis(hsi_to_rgb, axis=2, arr=hsi_img)

    return res 

image = Image.open('./images/color2.jpg')
img = np.array(image)
r,g,b = split_channels(img)
brightened = brighten(img, 20)
cntr = contrast(img, 1.5)
saturated = saturate(img, 1.5)

# Show images:
fig, axes = plt.subplots(2, 4, figsize=(20, 20))
axes = axes.ravel()

config = [
    (img, "Orignal Image"),
    (r, "Red channel"),
    (g, "Green channel"),
    (b, "Blue channel"),
    (brightened, "Brightened image"),
    (cntr, "Contrast image"),
    (saturated, "Saturated image")
]

for i, (image, title) in enumerate(config):
    axes[i].axis("off")
    axes[i].set_title(title)
    if "channel" in title:
        axes[i].imshow(image, cmap="gray")
    else:
        axes[i].imshow(image)

axes[-1].axis("off")

plt.show()