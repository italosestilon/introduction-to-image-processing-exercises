import numpy as np
from skimage import io, img_as_float, img_as_int
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.draw import rectangle_perimeter
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', help="Image's directory")
args = parser.parse_args()
img_path = args.image_dir


def load_image(dir):
    image = io.imread(dir)
    return np.logical_not(image.astype(np.bool))

def save_image(dir, image):
    image = np.logical_not(image).astype(np.uint8)
    cv2.imwrite(dir, image)

def dilation(image, s_elem):
    out = binary_dilation(image, s_elem)
    return out

def erosion(image, s_elem):
    out = binary_erosion(image, s_elem)
    return out

def closing(image, s_elem):
    out = binary_closing(image, s_elem)
    return out

def connected_components(image):
    image = image.copy()
    label_image = label(image)
    cc_number = label_image.max()

    proportion = np.zeros(cc_number)
    transitions_v = np.zeros(cc_number)
    transitions_h = np.zeros(cc_number, dtype=np.int32)

    for i, region in enumerate(regionprops(label_image)):
        minr, minc, maxr, maxc = region.bbox
        rr, cc = rectangle_perimeter(start=(minr, minc), end=(maxr, maxc), shape=image.shape)
        proportion[i] = image[minr:maxr, minc:maxc].sum()/((maxr-minr)*(maxc-minc))
        transitions_h[i] = np.count_nonzero(image[minr:maxr, minc: maxc-1] < image[minr:maxr, minc+1:maxc])
        #print(image[minr:maxr, minc: maxc-1])
        #transitions_v[i] = np.count_nonzero(image[])
        image[rr, cc] = True

    print(transitions_h)
    return image, cc_number, proportion

bitmap = load_image(args.image_dir)

bitmap_dilation_1 = dilation(bitmap, s_elem = np.ones((1, 100), dtype=np.uint8))
bitmap_erosion_1 = erosion(bitmap_dilation_1, s_elem = np.ones((1, 100), dtype=np.uint8))

bitmap_dilation_2 = dilation(bitmap, s_elem = np.ones((200, 1), dtype=np.uint8))
bitmap_erosion_2 = erosion(bitmap_dilation_2, s_elem = np.ones((200, 1), dtype=np.uint8))

bitmap_intersection = np.logical_and(bitmap_erosion_1, bitmap_erosion_2)

bitmap_closing = closing(bitmap_intersection, s_elem=np.ones((1, 30), dtype=np.uint8))

save_image('bitmap_dilation_1.pbm', bitmap_dilation_1)
save_image('bitmap_erosion_1.pbm', bitmap_erosion_1)


save_image('bitmap_dilation_2.pbm', bitmap_dilation_2)
save_image('bitmap_erosion_2.pbm', bitmap_erosion_2)

save_image('bitmap_intersection.pbm', bitmap_intersection)

save_image('bitmap_closing.pbm', bitmap_closing)


cc_bitmap, cc_number, proportion = connected_components(bitmap_closing)

save_image('cc_bitmap.pbm', cc_bitmap)

print(proportion)

#print(label_image.max())