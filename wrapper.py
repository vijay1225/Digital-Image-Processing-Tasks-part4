import numpy as np
from skimage.color import rgb2gray
import skimage.io as sk
from pathlib import Path
from scipy import signal
from matplotlib import pyplot
from skimage.filters import threshold_otsu
from skimage import transform
from allfunctions import *


def problem1_a():
    im1_path = Path('Data/Blurred-LowNoise.png')
    im2_path = Path('Data/Blurred-MedNoise.png')
    im3_path = Path('Data/Blurred-HighNoise.png')

    img1 = sk.imread(im1_path)
    img2 = sk.imread(im2_path)
    img3 = sk.imread(im3_path)

    img1_denoise = vijay_inverse_filtering(img1)
    img2_denoise = vijay_inverse_filtering(img2)
    img3_denoise = vijay_inverse_filtering(img3)

    pyplot.subplot(321)
    pyplot.title('Original')
    pyplot.imshow(img1, cmap='gray')
    pyplot.subplot(322)
    pyplot.title('Inverse Filtered Image')
    pyplot.imshow(np.log(np.abs(img1_denoise)), cmap='gray')
    pyplot.subplot(323)
    pyplot.imshow(img2, cmap='gray')
    pyplot.subplot(324)
    pyplot.imshow(np.log(np.abs(img2_denoise)), cmap='gray')
    pyplot.subplot(325)
    pyplot.imshow(img3, cmap='gray')
    pyplot.subplot(326)
    pyplot.imshow(np.log(np.abs(img3_denoise)), cmap='gray')
    pyplot.show()

    return

def problem1_b():
    im1_path = Path('Data/Blurred-LowNoise.png')
    im2_path = Path('Data/Blurred-MedNoise.png')
    im3_path = Path('Data/Blurred-HighNoise.png')

    img1 = sk.imread(im1_path)
    img2 = sk.imread(im2_path)
    img3 = sk.imread(im3_path)

    snr = 200

    img1_denoise = vijay_weiner_filtering(img1, snr=snr)
    img2_denoise = vijay_weiner_filtering(img2, snr=snr)
    img3_denoise = vijay_weiner_filtering(img3, snr=snr)

    pyplot.subplot(321)
    pyplot.title('Original')
    pyplot.imshow(img1, cmap='gray')
    pyplot.subplot(322)
    pyplot.title('Wiener filtering')
    pyplot.imshow(np.log(np.abs(img1_denoise)), cmap='gray')
    pyplot.subplot(323)
    pyplot.imshow(img2, cmap='gray')
    pyplot.subplot(324)
    pyplot.imshow(np.log(np.abs(img2_denoise)), cmap='gray')
    pyplot.subplot(325)
    pyplot.imshow(img3, cmap='gray')
    pyplot.subplot(326)
    pyplot.imshow(np.log(np.abs(img3_denoise)), cmap='gray')
    pyplot.show()


def problem1_c():
    im1_path = Path('Data/Blurred-LowNoise.png')
    im2_path = Path('Data/Blurred-MedNoise.png')
    im3_path = Path('Data/Blurred-HighNoise.png')

    img1 = sk.imread(im1_path)
    img2 = sk.imread(im2_path)
    img3 = sk.imread(im3_path)

    parameter = 0.005

    img1_denoise = vijay_least_square_filtering(img1, constraint_parameter=parameter)
    img2_denoise = vijay_least_square_filtering(img2, constraint_parameter=parameter)
    img3_denoise = vijay_least_square_filtering(img3, constraint_parameter=parameter)

    pyplot.subplot(321)
    pyplot.title('Original')
    pyplot.imshow(img1, cmap='gray')
    pyplot.subplot(322)
    pyplot.title('Least square filtering')
    pyplot.imshow(np.log(np.abs(img1_denoise)), cmap='gray')
    pyplot.subplot(323)
    pyplot.imshow(img2, cmap='gray')
    pyplot.subplot(324)
    pyplot.imshow(np.log(np.abs(img2_denoise)), cmap='gray')
    pyplot.subplot(325)
    pyplot.imshow(img3, cmap='gray')
    pyplot.subplot(326)
    pyplot.imshow(np.log(np.abs(img3_denoise)), cmap='gray')
    pyplot.show()
    return

def problem2_a():
    im1_path = Path('Data/noisy-book1.png')
    img1 = sk.imread(im1_path)
    image_size = np.shape(img1)
    sigma = 10 # sigma value low because i am doing smoothing in spatial domain with 5 x 5 window
    n = 5
    gaussian_filter_window = vijay_gaussian_filter_window(5, sigma=sigma)
    gaussian_filterd_image = signal.convolve2d(img1, gaussian_filter_window, boundary='symm', mode='same')
    meadian_filter_image = vijay_median_filtering(img1)

    pyplot.subplot(131)
    pyplot.title('Original')
    pyplot.imshow(img1, cmap='gray')
    pyplot.subplot(132)
    pyplot.title('Median filtered image')
    pyplot.imshow(meadian_filter_image, cmap='gray')
    pyplot.subplot(133)
    pyplot.title('Gaussian filtered image')
    pyplot.imshow(gaussian_filterd_image, cmap='gray')
    pyplot.show()
    return


def problem2_b():
    img1_path = Path('Data/noisy-book2.png')
    img1 = sk.imread(img1_path)
    sigma = 120  # here sigma so high because i am doing gaussina smoothing in frequency domain
    sigma_h = 100
    bilateral_image = vijay_bilateral_filtering(img1, sigma_h=sigma_h)
    gaussian_filtered_image = vijay_gaussian_filtering(img1, sigma=sigma)
    pyplot.subplot(131)
    pyplot.title('Original')
    pyplot.imshow(img1, cmap='gray')
    pyplot.subplot(132)
    pyplot.title('Smoothed with Bilatral Filter')
    pyplot.imshow(bilateral_image, cmap='gray')
    pyplot.subplot(133)
    pyplot.title('Smoothed with gaussian Filter')
    pyplot.imshow(gaussian_filtered_image, cmap='gray')
    pyplot.show()



def problem3():
    im1_path = Path('Data/barbara.tif')
    img1 = sk.imread(im1_path)

    image_size = np.shape(img1)
    output_size = [int(image_size[0] / 2), int(image_size[1] / 2)]
    without_smothing_downsampling = np.zeros(output_size)
    with_smothing_downsampling = np.zeros(output_size)
    smoothed_image = vijay_gaussian_filtering(img1, sigma=80)
    inbuilt_func_image = transform.resize(img1, output_size)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            without_smothing_downsampling[i, j] = img1[2 * i, 2 * j]
            with_smothing_downsampling[i, j] = smoothed_image[i * 2, j * 2]

    pyplot.subplot(141)
    pyplot.title('Original')
    pyplot.imshow(img1, cmap='gray')
    pyplot.subplot(142)
    pyplot.title('Without Smoothing Down sampling')
    pyplot.imshow(without_smothing_downsampling, cmap='gray')
    pyplot.subplot(143)
    pyplot.title('With Smoothing Down sampling')
    pyplot.imshow(with_smothing_downsampling, cmap='gray')
    pyplot.subplot(144)
    pyplot.title('Inbuilt function Down sampling')
    pyplot.imshow(inbuilt_func_image, cmap='gray')
    pyplot.show()



def problem4():
    im1_path = Path('Data/edge detection/grey50.png')
    im2_path = Path('Data/edge detection/grey51.png')
    im3_path = Path('Data/edge detection/grey52.png')
    img1 = sk.imread(im1_path)
    img2 = sk.imread(im2_path)
    img3 = sk.imread(im3_path)
    n = 5
    sigma = 10
    gaussian_filter_window = vijay_gaussian_filter_window(n, sigma)

    smooth_img1 = signal.convolve2d(img1, gaussian_filter_window, boundary='symm', mode='same')
    smooth_img2 = signal.convolve2d(img2, gaussian_filter_window, boundary='symm', mode='same')
    smooth_img3 = signal.convolve2d(img3, gaussian_filter_window, boundary='symm', mode='same')

    sobel_x = np.reshape([1, 0, -1, 2, 0, -2, 1, 0, -1], [3, 3])
    sobel_y = np.reshape([1, 2, 1, 0, 0, 0, -1, -2, -1], [3, 3])

    img1_edge_x = signal.convolve2d(smooth_img1, sobel_x, boundary='symm', mode='same')
    img2_edge_x = signal.convolve2d(smooth_img2, sobel_x, boundary='symm', mode='same')
    img3_edge_x = signal.convolve2d(smooth_img3, sobel_x, boundary='symm', mode='same')

    img1_edge_y = signal.convolve2d(smooth_img1, sobel_y, boundary='symm', mode='same')
    img2_edge_y = signal.convolve2d(smooth_img2, sobel_y, boundary='symm', mode='same')
    img3_edge_y = signal.convolve2d(smooth_img3, sobel_y, boundary='symm', mode='same')

    img1_edge_mag = np.sqrt((img1_edge_x ** 2) + (img1_edge_y ** 2))
    # th1 = threshold_otsu(img1_edge_mag)
    th1 = 20
    img1_edges_th = np.zeros(np.shape(img1_edge_mag))
    img1_edges_th[img1_edge_mag > th1] = 255

    img2_edge_mag = np.sqrt((img2_edge_x ** 2) + (img2_edge_y ** 2))
    # th2 = threshold_otsu(img2_edge_mag)
    th2 = 100
    img2_edges_th = np.zeros(np.shape(img2_edge_mag))
    img2_edges_th[img2_edge_mag > th2] = 255

    img3_edge_mag = np.sqrt((img3_edge_x ** 2) + (img3_edge_y ** 2))
    th3 = threshold_otsu(img3_edge_mag)
    th3 = 50
    img3_edges_th = np.zeros(np.shape(img3_edge_mag))
    img3_edges_th[img3_edge_mag > th3] = 255

    pyplot.subplot(231)
    pyplot.title('Original')
    pyplot.imshow(smooth_img1, cmap='gray')
    pyplot.subplot(232)
    pyplot.title('Original')
    pyplot.imshow(smooth_img2, cmap='gray')
    pyplot.subplot(233)
    pyplot.title('Original')
    pyplot.imshow(smooth_img3, cmap='gray')
    pyplot.subplot(234)
    pyplot.title('Edges')
    pyplot.imshow(img1_edges_th, cmap='gray')
    pyplot.subplot(235)
    pyplot.title('Edges')
    pyplot.imshow(img2_edges_th, cmap='gray')
    pyplot.subplot(236)
    pyplot.title('Edges')
    pyplot.imshow(img3_edges_th, cmap='gray')

    pyplot.show()


if __name__ == '__main__':
    # problem1_a()
    # problem1_b()
    problem1_c()
    # problem2_a()
    # problem2_b()
    # problem3()
    # problem4()
