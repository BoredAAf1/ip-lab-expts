import numpy as np

def fft(x):
    if len(x) == 1:
        return x
    
    n = len(x)
    
    even = fft(x[0::2])
    odd = fft(x[1::2])

    W = np.exp((-2j * np.pi * np.arange(n//2)) / n)

    return np.concatenate([even + W*odd, even - W*odd])

def fft_2d(X):
    X = X.astype(complex)
    M, N = X.shape

    # 1D fft along rows 
    for i in range(M):
        X[i, :] = fft(X[i, :])

    # 1D fft along columns
    for j in range(N):
        X[:, j] = fft(X[:, j])

    return X 

def ifft_2d(F):
    H, W = F.shape
    F_conj = np.conj(F)
    F_ift = fft_2d(F_conj)
    F_ift_conj = np.conj(F_ift)
    return np.real(F_ift_conj) / (H*W)

def shift(X):
    h, w = X.shape
    return np.roll(np.roll(X, h//2, axis=0), w//2, axis=1)

def unshift(X):
    h, w = X.shape
    return np.roll(np.roll(X, -(h//2), axis=0), -(w//2), axis=1)

def gaussian_lpf(h, w, d0):
    Y, X = np.ogrid[:h, :w]
    cy, cx = h//2, w//2
    u = X - cx 
    v = Y - cy 
    dist = u*u + v*v 
    return np.exp(-dist / (2*d0*d0))

def butterworth_lpf(h, w, d0, n):
    Y, X = np.ogrid[:h, :w]
    cy, cx = h//2, w//2
    u = X - cx 
    v = Y - cy 
    dist = np.sqrt(u*u + v*v)

    denom = 1 + (dist/d0)**(2*n)
    return 1/denom 

def ideal_lpf(h, w, d0):
    Y, X = np.ogrid[:h, :w]
    cy, cx = h//2, w//2
    u = X - cx 
    v = Y - cy 
    dist = u*u + v*v 
    return (dist <= d0**2)

def gaussian_hpf(h, w, d0):
    return 1 - gaussian_lpf(h, w, d0)

def laplacian_filter(h, w):
    Y, X = np.ogrid[:h, :w]
    cy, cx = h//2, w//2 
    u = X - cx 
    v = Y - cy 
    D2 = u*u + v*v + 1e-8
    return 10 * D2/np.max(D2) 

def filter_img(image):
    image = image.resize((128, 128))
    X = np.array(image, dtype=float)
    F = shift(fft_2d(X))
    H,W = X.shape 
    
    kernel = laplacian_filter(H, W)

    F *= kernel 

    X = ifft_2d(unshift(F))
    X = np.abs(X)
    X = np.clip(X, 0, 255).astype(np.uint8)

    return Image.fromarray(X)

from PIL import Image 

image = Image.open('./images/mario.png').convert('L')
filter_img(image).show()
