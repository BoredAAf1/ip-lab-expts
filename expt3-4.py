import numpy as np
from PIL import Image

def linear_filtering(image, kernel):
    k = kernel.shape[0]
    p = k // 2

    padded = np.pad(image, ((p, p), (p, p)), mode='edge')

    out = np.zeros_like(image)
    h, w = out.shape

    for y in range(h):
        for x in range(w):
            region = padded[y:y+k, x:x+k]
            out[y, x] = np.sum(kernel * region)

    return out

def mean_filter(k):
    kernel = np.ones((k, k)) / (k*k)
    return kernel 

def gaussian_filter(k, s=1.0):
    kernel = np.zeros((k, k))
    p = k//2 

    for x in range(-p, p+1):
        for y in range(-p, p+1):
            kernel[x+p, y+p] = 1/(2*np.pi*s*s) * np.exp(-(x*x + y*y)/(2*s*s))
    
    kernel /= kernel.sum()
    
    return kernel

def nonlinear_filtering(image, k, operation):
    p = k//2
    padded = np.pad(image, ((p, p), (p, p)), mode='edge')

    out = np.zeros_like(image)

    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            region = padded[y:y+k, x:x+k]
            out[y, x] = operation(region)
    
    return out.astype(np.uint8)

img = Image.open('./images/mario.png').convert('L')
img = np.array(img).astype(float)
# Linear, Nonlinear filtering
filtered = linear_filtering(img, mean_filter(7))
filtered = nonlinear_filtering(img, 3, np.median)
Image.fromarray(filtered.astype(np.uint8)).show()


# Sobel edge detection
sobelH = np.array([[1, 0, -1], 
                   [2, 0, -2], 
                   [1, 0, -1]])
sobelV = sobelH.T 
edge_h = linear_filtering(img, sobelH)
edge_v = linear_filtering(img, sobelV)
edge = np.sqrt(edge_h ** 2 + edge_v ** 2).astype(np.uint8)
Image.fromarray(edge).show()

# Prewitt edge detection
prewittH = np.array([[1, 0, -1], 
                [1, 0, -1], 
                [1, 0, -1]])
prewittV = prewittH.T 
edge_h = linear_filtering(img, prewittH)
edge_v = linear_filtering(img, prewittV)
edge = np.sqrt(edge_h ** 2 + edge_v ** 2).astype(np.uint8)
Image.fromarray(edge).show()

# Laplacian edge detection
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
lap2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
edge = linear_filtering(img, lap2)
edge = np.abs(edge).astype(np.uint8)
Image.fromarray(edge).show()