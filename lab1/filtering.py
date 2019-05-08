import numpy as np
from scipy import ndimage
from skimage import io, img_as_float, img_as_int
from scipy import signal
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

def fft(x, shape=None):
  if shape == None:
    shape = x.shape

  fft_x = np.fft.fftn(x, shape)
  fft_x_shifted = np.fft.fftshift(fft_x)

  return fft_x_shifted


def fft_convolve(image, kernel):
  shape = image.shape
  
  fft_image_shifted = fft(image, shape)
  fft_kernel_shifted =fft(kernel, shape)

  fft_convolve_shifted = fft_image_shifted * fft_kernel_shifted
  
  fft_convolve = np.fft.ifftshift(fft_convolve_shifted)

  filtered = np.fft.ifftn(fft_convolve)
  
  return filtered


def filter_with_gaussian(image, kernel_size = 5, sigma = 1):
    gaussian_signal = signal.gaussian(kernel_size, sigma)
    kernel = np.outer(gaussian_signal, gaussian_signal)

    kernel = kernel/kernel.sum()

    return fft_convolve(image, kernel)

def normalize_image(image):
  f = image
  f = 1*(f - f.min())/(f.max() - f.min())

  return f
  

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
filtered_h3_h4 = normalize_image(filtered_h3_h4)

io.imsave('filtered_h1.png', image_filtered_h1)
io.imsave('filtered_h2.png', image_filtered_h2) 
io.imsave('filtered_h3.png', image_filtered_h3)
io.imsave('filtered_h4.png', image_filtered_h4)
io.imsave('filtered_h3_h4.png', filtered_h3_h4)

shifted = fft(image)
shifted = np.abs(shifted)
norm = LogNorm()
#normalize in logaritmic scale to plot
shifted = norm(shifted)
io.imsave('fourier_spectre.png', shifted)

for i in range(1, 7):
    filtered = filter_with_gaussian(image, kernel_size=5, sigma=i)

    img_name_to_save = "filtered_gaussian_{}.png".format(i)

    filtered = np.abs(filtered)
    filtered = normalize_image(filtered)

    io.imsave(img_name_to_save, filtered)