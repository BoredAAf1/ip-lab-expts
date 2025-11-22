import numpy as np
from PIL import Image

def fft(x):
    n = len(x)
    if n == 1:
        return x
    
    even = fft(x[0::2])
    odd = fft(x[1::2])

    W = np.exp((-2j * np.pi * np.arange(n//2)) / n )

    return np.concatenate([even + W*odd, even - W*odd])

def fft_2d(X):
    X = X.copy().astype(complex)
    h, w = X.shape

    for i in range(h):
        X[i, :] = fft(X[i, :])
    
    for j in range(w):
        X[:, j] = fft(X[:, j])
    
    return X 

def ifft_2d(X):
    h, w = X.shape
    return np.real(np.conj(fft_2d(np.conj(X))) / (h*w))

def shift(F):
    h, w = F.shape
    return np.roll(np.roll(F, h//2, axis=0), w//2, axis=1)

def unshift(F):
    h, w = F.shape
    return np.roll(np.roll(F, -(h//2), axis=0), -(w//2), axis=1)

def gaussian_lpf(H, W, D0):
    Y, X = np.ogrid[:H, :W]
    u = X - W//2
    v = Y - H//2 
    dist = u**2 + v**2
    return np.exp( - dist / (2*D0*D0))

def gaussian_hpf(H, W, D0):
    return 1 - gaussian_lpf(H, W, D0)
                            
def freq_filter(image, D0):
    image = Image.open(image).convert('L')
    image = image.resize((256, 256))
    X = np.array(image).astype(float)

    F = shift(fft_2d(X))
    H, W = F.shape 
    mask = gaussian_hpf(H, W, D0)

    F = F*mask 
    X = ifft_2d(unshift(F))
    X = np.clip(X, 0, 255).astype(np.uint8)

    return Image.fromarray(X)

filtered_img = freq_filter("images/mario.png", D0=20)
filtered_img.show()