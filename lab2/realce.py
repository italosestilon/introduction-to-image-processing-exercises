import numpy as np
from skimage import io, img_as_float, img_as_int
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', help="Image's directory to filter")
args = parser.parse_args()

def load_image(dir):
    image = io.imread(dir)
    return image


def normalize_image(image, n=9):
  f = image.copy().astype(np.float)
  print(n)
  print(f.max() - f.min())
  f = n*(f - f.min())/(f.max() - f.min())
  print(f.max())
  return f

def dithering(f, mask, pattern, inverse_line=False, pattern_i=None):
    
    f = img_as_float(f.copy())
    g = np.zeros_like(f, dtype=float)
    n = f.shape[0]
    m = f.shape[1]
    print(f.shape)

    for i in range(n):
        line_star, line_end, pace = (0, m, 1) if (not inverse_line) or (i % 2 != 0) else (m-1, 0, -1)
        for j in range(line_star, line_end, pace):
            if(f[i,j] > 0.5):
                g[i, j] = 1
            else:
                g[i, j] = 0
            error = f[i, j] - g[i, j]
            for (move_i, move_j), w in zip(pattern, mask):
                move_i += i
                move_j += j
                if(move_i < n and move_j < m):
                    f[move_i, move_j] += (error*w)
    return g


def ordered_dithering(image, mask, l_max):

    f = image.copy()
    f = normalize_image(f, 9)
    print(image.max())
    print(image.min())
    print(f.max())
    n = f.shape[0]
    m = f.shape[1]

    g = np.zeros_like(f, np.float)

    for i in range(n):
        for j in range(m):
            th = mask[i%3, j%3]
            g[i, j] = 0 if f[i, j] < th else 1
    
    return g


image = load_image(args.image_dir)

#print(image.astype(np.float).max())
mask = [7.0/16, 3.0/16, 5.0/16, 1.0/16]
pattern = [(0, 1), (1, -1), (1, 0), (1, 1)]
g = dithering(image, mask, pattern, inverse_line=True)
io.imsave('error_difusion.png', g)

bayer_mask = np.array([[6,8,4], [1,0,3], [5,2,7]])
g_bayer = ordered_dithering(image, bayer_mask, 9)
io.imsave('bayer_mask.png', g_bayer)

dispersed_mask = np.array([[1, 7, 8], [8, 5, 3], [6, 2, 9]])
g_dispersed = ordered_dithering(image, dispersed_mask, 9)
io.imsave('dispersed_mask.png', g_dispersed)

#cv2.imwrite('dispersed_mask.pbm', g_dispersed)
Image.fromarray(normalize_image(g_dispersed, 255).astype(np.uint8)).save('g.pbm')