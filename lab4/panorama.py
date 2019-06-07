#%%
import numpy as np
from skimage import io, img_as_float, img_as_int
import cv2
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('image_dir', help="Image's directory")
#args = parser.parse_args()
#img_path = args.image_dir

#%%
def load_image(dir):
    image = io.imread(dir)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image


def save_image(dir, image):
    cv2.imwrite(dir, image)


def apply_sift(image):
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detect(image, None)

    return key_points


#%%
image1_dir = 'images/foto1A.jpg'
image2_dir = 'images/foto1B.jpg'
image1, gray_image1 = load_image(image1_dir)
image2, gray_image2 = load_image(image2_dir)

#%%
key_points = apply_sift(gray_image1)

#%%

sift_image1 = cv2.drawKeypoints(gray_image1,key_points,image1)
#%%
io.imshow(sift_image1)