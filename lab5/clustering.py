
import numpy as np
from skimage import io, img_as_float, img_as_int
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image', help="Image1's directory")
args = parser.parse_args()

def load_image(dir):
    image = io.imread(dir)
    return img_as_float(image)

def save_image(dir, image):
    io.imsave(dir, image)

def clustering(image, num_clusters):
    image_shape = image.shape
    image_as_points = image.reshape(image_shape[0] * image_shape[1], 3)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(image_as_points)
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(image_as_points)

    new_image = centers[predict]
    new_image = new_image.reshape(image_shape)

    return new_image


image_dir = args.image
image_name = image_dir.split('/')[-1]

image = load_image(image_dir)

new_image = clustering(image, 5)

save_image('new_image.png', new_image)


