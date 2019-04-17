import numpy as np
from scipy import ndimage
from skimage import io, img_as_ubyte, img_as_float, img_as_int
from skimage import transform
from scipy import signal
from skimage.filters import sobel_h, sobel_v
from matplotlib.colors import LogNorm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', help="Image's directory to filter")
args = parser.parse_args()

def load_image(dir):
    image = io.imread(dir)
    return img_as_int(image)

def filter(image, kernel):
    image_filtered = ndimage.convolve(image, kernel, mode='constant', cval=0)
    return image_filtered

def fft_convolve(image, kernel):
  shape = np.array(image.shape)
  fft_image = np.fft.fftn(image, shape)
  fft_image_shifted = np.fft.fftshift(fft_image)
  fft_kernel = np.fft.fftn(kernel, shape)
  fft_kernel_shifted = np.fft.fftshift(fft_kernel)

  fft_convolve_shifted = fft_image_shifted * fft_kernel_shifted
  
  fft_convolve = np.fft.ifftshift(fft_convolve_shifted)

  filtered = np.fft.ifftn(fft_convolve)
  
  return filtered, fft_image_shifted


def filter_with_gaussian(image, kernel_size = 5, sigma = 1):
    gaussian_signal = signal.gaussian(kernel_size, sigma)
    kernel = np.outer(gaussian_signal, gaussian_signal)

    kernel = kernel/kernel.sum()

    return fft_convolve(image, kernel)

h1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
h2 = (1.0/256.0) * np.array([[1, 4, 6 , 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6 , 4, 1]])
h3 = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])
h4 = h3.transpose()

high = np.array([[0,-1, 0], [-1, 4, -1], [0, -1, 0]])

image = load_image(args.image_dir)

image_filtered_h3 = filter(image, h3)
image_filtered_h2 = filter(image, h2)
image_filtered_h1 = filter(image, high)
image_filtered_h4 = filter(image, h4)
filtered_h3_h4 = np.sqrt(img_as_float(image_filtered_h3)**2 + img_as_float(image_filtered_h4)**2)

#normilize filtered_h3_h4
f = filtered_h3_h4
f = 1*(f - f.min())/(f.max() - f.min())

io.imsave('filtered_h1.png', image_filtered_h1)
io.imsave('filtered_h2.png', image_filtered_h2) 
#io.imsave('h1_after_h2.png', h1_after_h2)
io.imsave('filtered_h3.png', image_filtered_h3)
io.imsave('filtered_h4.png', image_filtered_h4)
io.imsave('filtered_h3_h4.png', f)

for i in range(1, 7):
    filtered, shifted = filter_with_gaussian(image, kernel_size=5, sigma=i)

    if(i == 1):
        shifted = np.abs(shifted)
        norm = LogNorm()
        shifted = norm(shifted)

        io.imsave('fourier_spectre.png', shifted)

    img_name_to_save = "filtered_gaussian_{}.png".format(i)

    filtered = np.abs(filtered)
    f = filtered
    f = 1*(f - f.min())/(f.max() - f.min())

    io.imsave(img_name_to_save, f)