
import numpy as np
from skimage import io, img_as_float, img_as_int
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image1', help="Image1's directory")
parser.add_argument('image2', help="Image2's directory")
args = parser.parse_args()

image1_dir = args.image1
image2_dir = args.image2
image1_name = image1_dir.split('/')[-1]
image2_name = image2_dir.split('/')[-1]

def load_image(dir):
    image = cv2.imread(dir)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
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

def apply_brief(image):
    kps, _= apply_sift(image)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    key_points, descriptors = brief.compute(image, kps)

    return key_points, descriptors

def apply_orb(image):
    orb = cv2.ORB_create()
    key_points = orb.detect(image, None)

    key_points, descriptors = orb.compute(image, key_points)
    return key_points, descriptors

def calculate_match(descriptors1, descriptors2, norm=cv2.NORM_L1, cross_check=False):
    bf_matcher = cv2.BFMatcher_create(normType=norm, crossCheck=cross_check)
    raw_matches = bf_matcher.knnMatch(descriptors1, descriptors2, 2)
    matches = bf_matcher.match(descriptors1, descriptors2)
    #matches = []
    #for match in raw_matches:
    #    if(match[0].distance < match[1].distance*0.75):
    #        matches.append((match[0].trainIdx, match[0].queryIdx, match[0].distance))

    matches.sort(key= lambda elem: elem.distance)

    return matches

def find_homography_matrix(keypoints1, keypoints2, matches):
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float)
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float)
    M, mask = cv2.findHomography(keypoints1, keypoints2, method=cv2.RANSAC, ransacReprojThreshold=4.0)

    return M, mask

def draw_lines(image1, keypoints1, image2, keypoints2, matches):
    image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, outImg=None, flags=2)
    return image

def align_and_join_images(image1, image2, homography_matrix):
    h_1, w_1, _ = image1.shape
    h_2, w_2, _ = image2.shape
    result_image = cv2.warpPerspective(image1, homography_matrix, (w_1+w_2, h_2))
    result_image[0:h_2, 0:w_2] = image2
    return result_image

def panoramica(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2, alg, image_name):
    matches = calculate_match(descriptors1, descriptors2)
    image_with_lines = draw_lines(
        image1.copy(), keypoints1, image2.copy(), keypoints2, matches[:45])

    save_image('out/{}_with_lines_{}.jpg'.format(image_name, alg), image_with_lines)

    homography_matrix, mask = find_homography_matrix(
        keypoints1, keypoints2, matches[:45])
    print(len(matches))
    aligned_image = align_and_join_images(image1, image2, homography_matrix)
    save_image('out/paranoramica_{}_{}.jpg'.format(image_name, alg), aligned_image)

def sift(image1, image2, gray_image1, gray_image2, image1_name, image2_name):
    key_points, descriptors = apply_sift(gray_image1)

    key_points2, descriptors2 = apply_sift(gray_image2)

    panoramica(image1, image2, key_points, key_points2,
            descriptors, descriptors2, alg='SIFT', image_name=image1_name[:5])
    sift_image1 = cv2.drawKeypoints(gray_image1, key_points, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sift_image2 = cv2.drawKeypoints(
        gray_image2, key_points2, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    save_image('out/{}_sift.jpg'.format(image1_name), sift_image1)
    save_image('out/{}_sift.jpg'.format(image2_name), sift_image2)

def surf(image1, image2, gray_image1, gray_image2, image1_name, image2_name):
    keypoints1, descriptors1 = apply_surf(gray_image1, threshold=10000)
    keypoints2, descriptors2 = apply_surf(gray_image2, threshold=10000)

    surf_image1 = cv2.drawKeypoints(gray_image1, keypoints1, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    surf_image2 = cv2.drawKeypoints(gray_image2, keypoints2, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    save_image('out/{}_surf.jpg'.format(image1_name), surf_image1)
    save_image('out/{}_surf.jpg'.format(image2_name), surf_image2)

    panoramica(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2, alg='SURF', image_name=image1_name[:5])


def brief(image1, image2, gray_image1, gray_image2, image1_name, image2_name):
    keypoints1, descriptors1 = apply_brief(gray_image1)
    keypoints2, descriptors2 = apply_brief(gray_image2)

    brief_image1 = cv2.drawKeypoints(gray_image1, keypoints1, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    brief_image2 = cv2.drawKeypoints(gray_image2, keypoints2, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    save_image('out/{}_brief.jpg'.format(image1_name), brief_image1)
    save_image('out/{}_brief.jpg'.format(image2_name), brief_image2)

    panoramica(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2, alg='BRIEF', image_name=image1_name[:5])

def orb(image1, image2, gray_image1, gray_image2, image1_name, image2_name):
    keypoints1, descriptors1 = apply_orb(gray_image1)
    keypoints2, descriptors2 = apply_orb(gray_image2)

    orb_image1 = cv2.drawKeypoints(gray_image1, keypoints1, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    orb_image2 = cv2.drawKeypoints(gray_image2, keypoints2, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    save_image('out/{}_brief.jpg'.format(image1_name), orb_image1)
    save_image('out/{}_surf.jpg'.format(image2_name), orb_image2)

    panoramica(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2, alg='ORB', image_name=image1_name[:5])

image1, gray_image1 = load_image(image1_dir)
image2, gray_image2 = load_image(image2_dir)

sift(image1, image2, gray_image1, gray_image2, image1_name, image2_name)

surf(image1, image2, gray_image1, gray_image2, image1_name, image2_name)

brief(image1, image2, gray_image1, gray_image2, image1_name, image2_name)

orb(image1, image2, gray_image1, gray_image2, image1_name, image2_name)
