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
    transitions_v = np.zeros(cc_number, dtype=np.float)
    transitions_h = np.zeros(cc_number, dtype=np.float)

    for i, region in enumerate(regionprops(label_image)):
        minr, minc, maxr, maxc = region.bbox
        rr, cc = rectangle_perimeter(
            start=(minr, minc), end=(maxr, maxc), shape=image.shape)
        pixel_numbers = (maxr-minr)*(maxc-minc)
        proportion[i] = image[minr:maxr, minc:maxc].sum()/(pixel_numbers)

        transitions_h[i] = np.float(np.count_nonzero(
            image[minr:maxr, minc: maxc-1] < image[minr:maxr, minc+1:maxc]))/pixel_numbers
        transitions_v[i] = np.float(np.count_nonzero(np.transpose(
            image[minr:maxr, minc: maxc-1]) < np.transpose(image[minr:maxr, minc+1:maxc])))/pixel_numbers

        image[rr, cc] = True

    return label_image, image, cc_number, proportion, transitions_h, transitions_v

def segment_text(label, image, proportion, transitions_h, transitions_v):
    image = image.copy()
    text_regions = []
    for i, region in enumerate(regionprops(label)):
        if((proportion[i] > 0.45 and proportion[i] < 0.97) and (transitions_h[i] <= 0.05 and transitions_v[i] < 0.05)):
            minr, minc, maxr, maxc = region.bbox
            rr, cc = rectangle_perimeter(
                start=(minr, minc), end=(maxr, maxc), shape=image.shape)
            image[rr, cc] = True
            text_regions.append((minr, minc, maxr, maxc))
    return image, text_regions

def draw_retangles(image, label_image):
    image = image.copy()

    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        rr, cc = rectangle_perimeter(
            start=(minr, minc), end=(maxr, maxc), shape=image.shape)
        image[rr, cc] = True

    return image

def segment_words(text_regions, image):
    image = image.copy()
    _image = np.full_like(image, False, dtype=np.bool)
    for region in text_regions:
        minr, minc, maxr, maxc = region
        _image[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]

    save_image('image_only_text.pbm', _image)

    _image_dilation = dilation(_image, s_elem=np.ones((8, 12), dtype=np.uint8))
    _image_erosion = erosion(_image_dilation, s_elem=np.ones((8, 12), dtype=np.uint8))

    label_image, _, _, _, _, _ = connected_components(_image_erosion)

    _image_with_retangles = draw_retangles(image, label_image)

    number_of_words = label_image.max()
    return _image_with_retangles, number_of_words

def count_lines(text_regions, image):
    image = image.copy()
    _image = np.full_like(image, False, dtype=np.bool)
    for region in text_regions:
        minr, minc, maxr, maxc = region
        _image[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]
    
    _image_closing = closing(_image, s_elem=np.ones((8, 130), dtype=np.uint8))

    save_image('lines.pbm', _image_closing)

    label_image, number_of_lines = label(_image_closing, return_num=True)
    
    lines_segmented = draw_retangles(image, label_image)
    save_image('lines_segmented.pbm', lines_segmented)

    return number_of_lines

bitmap = load_image(args.image_dir)

#step 1
bitmap_dilation_1 = dilation(bitmap, s_elem=np.ones((1, 100), dtype=np.uint8))
save_image('bitmap_dilation_1.pbm', bitmap_dilation_1)

#step 2
bitmap_erosion_1 = erosion(
    bitmap_dilation_1, s_elem=np.ones((1, 100), dtype=np.uint8))
save_image('bitmap_erosion_1.pbm', bitmap_erosion_1)

#step 3
bitmap_dilation_2 = dilation(bitmap, s_elem=np.ones((200, 1), dtype=np.uint8))
save_image('bitmap_dilation_2.pbm', bitmap_dilation_2)

#step 4
bitmap_erosion_2 = erosion(
    bitmap_dilation_2, s_elem=np.ones((200, 1), dtype=np.uint8))
save_image('bitmap_erosion_2.pbm', bitmap_erosion_2)

#step 5
bitmap_intersection = np.logical_and(bitmap_erosion_1, bitmap_erosion_2)
save_image('bitmap_intersection.pbm', bitmap_intersection)

#step 6
bitmap_closing = closing(
    bitmap_intersection, s_elem=np.ones((1, 30), dtype=np.uint8))
save_image('bitmap_closing.pbm', bitmap_closing)

#step 7
label_image, cc_bitmap, cc_number, proportion, transitions_h, transitions_v = connected_components(
    bitmap_closing)

save_image('cc_bitmap.pbm', cc_bitmap)

#step 8 and 9
bitmap_segmented, text_regions = segment_text(label_image, bitmap, proportion, transitions_h, transitions_v)
save_image('image_segmented.pbm', bitmap_segmented)

#step 10
words_segmented, number_of_words = segment_words(text_regions, bitmap)
save_image('words_segmented.pbm', words_segmented)
number_of_lines = count_lines(text_regions, bitmap)

print("Number of lines {}".format(number_of_lines))
print("Number of words {}".format(number_of_words))
