import numpy as np
from skimage import io, img_as_float, img_as_int
from skimage.morphology import binary_dilation
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', help="Image's directory")
args = parser.parse_args()
img_path = args.image_dir


def load_image(dir):
    image = io.imread(dir)
    return image

def dilation(image):
    s_elem = np.ones((1, 100), dtype=np.uint8)
    out = binary_dilation(image, s_elem)
    return out

bitmap = load_image(args.image_dir)
bitmap_dilation = dilation(bitmap)
cv2.imwrite('bitmap_dilation.pbm', bitmap_dilation.astype(np.uint8))