import numpy as np
from scipy import ndimage
from skimage import io, img_as_ubyte, img_as_float, img_as_int
from skimage import transform
from scipy.fftpack import ifft2, fft2, fftshift

from skimage.filters import sobel_h, sobel_v

import cv2

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

def discrete_fourier_transform(image):
    transformed = fft2(image)
    transformed = fftshift(transformed)
    return transformed

h1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
h2 = (1.0/256.0) * np.array([[1, 4, 6 , 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6 , 4, 1]])
h3 = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])
h4 = h3.transpose()

high = np.array([[0,-1, 0], [-1, 4, -1], [0, -1, 0]])

image = load_image(args.image_dir)

image_filtered_h3 = filter(image, h3)
image_filtered_h2 = filter(image, h2)
image_filtered_h1 = filter(image, high)
#h1_after_h2 = filter(image_filtered_h2, h1)
image_filtered_h4 = filter(image, h4)
filtered_h3_h4 = np.sqrt(img_as_float(image_filtered_h3)**2 + img_as_float(image_filtered_h4)**2)

print(filtered_h3_h4.max())
#print('shape', image.shape)

#print(image_filtered_h2.max())
#print(np.sqrt(image_filtered_h3**2 + image_filtered_h4**2))

#f = np.absolute(image_filtered_h3)
#print()
f = filtered_h3_h4
f = 1*(f - f.min())/(f.max() - f.min())

io.imsave('filtered_h1.png', image_filtered_h1)
io.imsave('filtered_h2.png', image_filtered_h2) 
#io.imsave('h1_after_h2.png', h1_after_h2)
io.imsave('filtered_h3.png', image_filtered_h3)
io.imsave('filtered_h4.png', image_filtered_h4)
io.imsave('filtered_h3_h4.png', f)

fft_image = discrete_fourier_transform(image)

#print(np.real(fft_image))

#io.imsave('fft_image.png', (np.real(fft_image)/np.real(fft_image).max()))