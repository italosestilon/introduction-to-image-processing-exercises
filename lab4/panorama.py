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
    key_points, descriptors = sift.detectAndCompute(image, None)

    return key_points, descriptors

def apply_surf(image, threshold=400):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=threshold)
    key_points, descriptors = surf.detectAndCompute(image, None)

    return key_points, descriptors

def apply_brief(image, key_points):
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    key_points, descriptors = brief.compute(image, key_points)

    return key_points, descriptors

def apply_orb(image):
    orb = cv2.ORB_create()
    key_points = orb.detect(image, None)

    key_points, descriptors = orb.compute(image, key_points)
    return key_points, descriptors
#%%
image1_dir = 'images/foto1A.jpg'
image2_dir = 'images/foto1B.jpg'
image1, gray_image1 = load_image(image1_dir)
image2, gray_image2 = load_image(image2_dir)

#%%
key_points = apply_sift(gray_image1)

#%%

sift_image1 = cv2.drawKeypoints(gray_image1,key_points,image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#%%
io.imshow(sift_image1)

#%%

surf_keypoints, suft_descriptors = apply_surf(gray_image1, 5000)
surf_image1 = cv2.drawKeypoints(gray_image1, surf_keypoints, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#%%
io.imshow(surf_image1)

#%%
kp, des = apply_brief(image1, surf_keypoints)
print(des.shape)
#%%

orb_keypoints, orb_descriptors = apply_orb(gray_image1)
orb_image1 = cv2.drawKeypoints(gray_image1, orb_keypoints, None,
                               (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


io.imshow(orb_image1)

#%%
