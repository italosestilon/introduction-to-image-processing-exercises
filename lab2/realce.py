import numpy as np
from skimage import io, img_as_float, img_as_int
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', help="Image's directory to filter")
args = parser.parse_args()

def load_image(dir):
    image = io.imread(dir)
    return image


def dithering(f, mask, pattern):
    
    f = img_as_float(f.copy())
    g = np.zeros_like(f, dtype=float)
    n = f.shape[0]
    m = f.shape[1]
    print(f.shape)

    for i in range(n):
        for j in range(m):

            if(f[i,j] > 0.5):
                g[i, j] = 1
            else:
                g[i, j] = 0

            error = f[i, j] - g[i, j]
            for (move_i, move_j), w in zip(pattern, mask):
                move_i += i
                move_j += j
                if(move_i < n and move_j < m and j >= 0):
                    f[move_i, move_j] += (error*w)
    return g
image = load_image(args.image_dir)
#print(image.astype(np.float).max())
mask = [7.0/16, 3.0/16, 5.0/16, 1.0/16]
pattern = [(0, 1), (1, -1), (1, 0), (1, 1)]
g = dithering(image, mask, pattern)
print(g)
io.imsave('error_difusion.png', g)
