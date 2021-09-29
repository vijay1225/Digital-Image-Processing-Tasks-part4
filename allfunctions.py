import numpy as np
from skimage.color import rgb2gray
import skimage.io as sk
from pathlib import Path
from scipy import signal
from matplotlib import pyplot
from skimage.filters import threshold_otsu
from scipy.io import loadmat
from skimage import filters


def vijay_gaussian_filter_window(n,m=0, sigma = 10):
    if m == 0:
        m = n
    gaussian_filter = np.zeros([n, m])
    if n%2 == 0:
        l = int(n/2) - 1
    else:
        l = np.int(np.floor(n / 2))
    if m%2 == 0:
        u = int(m/2) - 1
    else:
        u = np.int(np.floor(m/2))
    for i in range(-l, l+1):
        for j in range(-u, u+1):
            gaussian_filter[i + l, j + u] = np.exp(-((i ** 2) + (j ** 2)) / (sigma ** 2))
    return gaussian_filter / np.sum(gaussian_filter)


def vijay_gaussian_filtering(input_image, sigma = 100):
    dft_character_image = np.fft.fftshift(np.fft.fft2(input_image))
    size = np.shape(dft_character_image)
    gaussian_filter = vijay_gaussian_filter_window(size[0],size[1],sigma)
    dft_filtered_image = dft_character_image * gaussian_filter
    filtered_image = np.fft.ifft2(np.fft.fftshift(dft_filtered_image))

    return np.absolute(filtered_image)


def vijay_bilateral_filtering(input_image, sigma_h = 100):
    image_size = np.shape(input_image)
    l = image_size[0]
    w = image_size[1]
    img1_zero_padded = np.zeros([l + 3, w + 3])
    img1_denoise_bilatral = np.zeros(image_size)
    img1_zero_padded[3:l + 3, 3:w + 3] = input_image.copy()
    gaussian_filter = vijay_gaussian_filter_window(7,sigma=10)
    for i in range(3, l):
        for j in range(3, w):
            image_window = img1_zero_padded[i - 3:i + 4, j - 3:j + 4]
            h = np.exp((-((img1_zero_padded[i, j] * np.ones([7, 7])) - image_window) ** 2) / (2 * sigma_h ** 2))
            h = h / np.sum(h)
            k_window = gaussian_filter * h
            k = np.sum(k_window)
            img1_denoise_bilatral[i, j] = np.sum(image_window * k_window) / k
    return img1_denoise_bilatral


def vijay_median_filtering(input_image):
    image_size = np.shape(input_image)
    meadian_filter_image = np.zeros(image_size)
    zero_padded_image = np.zeros([image_size[0] + 2, image_size[1] + 2])
    zero_padded_image[1:image_size[0] + 1, 1:image_size[1] + 1] = input_image
    for i in range(1, image_size[0]):
        for j in range(1, image_size[1]):
            test = np.ravel(zero_padded_image[i - 1:i + 2, j - 1:j + 2])
            test = np.sort(test)
            meadian_filter_image[i, j] = test[4]
    return meadian_filter_image


def vijay_inverse_filtering(input_image):
    image_size = np.shape(input_image)

    kernal1 = loadmat('Data/BlurKernel.mat')
    kernal1_size = np.shape(kernal1['h'])
    kernal = np.zeros(image_size)
    kernal[0:kernal1_size[0], 0:kernal1_size[1]] = kernal1['h']
    img1_dft = np.fft.fftshift(np.fft.fft2(input_image))
    kernal_dft = np.fft.fftshift(np.fft.fft2(kernal))

    img1_denoise_dft = img1_dft / (kernal_dft)
    return np.fft.ifft2(np.fft.fftshift(img1_denoise_dft))


def vijay_weiner_filtering(input_image, snr):
    image_size = np.shape(input_image)

    kernal1 = loadmat('Data/BlurKernel.mat')
    kernal1_size = np.shape(kernal1['h'])
    kernal = np.zeros(image_size)
    kernal[0:kernal1_size[0], 0:kernal1_size[1]] = kernal1['h']
    img1_dft = np.fft.fftshift(np.fft.fft2(input_image))
    kernal_dft = np.fft.fftshift(np.fft.fft2(kernal))
    temp1 = 1/snr
    temp2 = temp1 + (kernal_dft * np.conjugate(kernal_dft))
    d = np.conjugate(kernal_dft) / temp2
    img1_dft_denoise = img1_dft * d
    img1_denoise = np.fft.ifft2(np.fft.fftshift(img1_dft_denoise))
    return img1_denoise


def vijay_least_square_filtering(input_image, constraint_parameter = 0.005):
    image_size = np.shape(input_image)
    gamma = constraint_parameter
    kernal1 = loadmat('Data/BlurKernel.mat')
    kernal1_size = np.shape(kernal1['h'])
    kernal = np.zeros(image_size)
    kernal[0:kernal1_size[0], 0:kernal1_size[1]] = kernal1['h']
    p1 = np.reshape([0, -1, 0, -1, 4, -1, 0, -1, 0], [3, 3])
    p = np.zeros(image_size)
    p[0:3, 0:3] = p1
    img1_dft = np.fft.fftshift(np.fft.fft2(input_image))
    kernal_dft = np.fft.fftshift(np.fft.fft2(kernal))
    p_dft = np.fft.fftshift(np.fft.fft2(p))
    temp1 = (kernal_dft * np.conjugate(kernal_dft)) + (gamma * (p_dft * np.conjugate(p_dft)))
    d = np.conjugate(kernal_dft) / temp1
    denoise_image_dft = img1_dft * d
    return np.fft.ifft2(np.fft.fftshift(denoise_image_dft))