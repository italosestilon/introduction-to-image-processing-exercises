
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float, img_as_int
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('image', help="Image1's directory")
args = parser.parse_args()

def load_image(dir):
    image = io.imread(dir)
    return img_as_float(image)

def save_image(dir, image):
    io.imsave(dir, image)

def visualize_image(image, image_name):
    image_shape = image.shape
    image_as_points = image.reshape(image_shape[0] * image_shape[1], 3)

    pca = PCA(n_components=2)
    pca.fit(image_as_points)

    image_as_points_reduced = pca.transform(image_as_points)
    print(image_as_points_reduced[:,1:2].shape)
    x = image_as_points_reduced[:, 0:1].flatten()
    y = image_as_points_reduced[:, 1:2].flatten()
    print(x)
    print(y)
    plt.scatter(x, y, s=10, c='r', marker='o')
    plt.savefig('out/viz_{}'.format(image_name))

def plot_cluster(X, centers, labels, k, image_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pca = PCA(n_components=2)
    X_r =pca.fit_transform(X)
    x = X_r[:,0:1].flatten()
    y = X_r[:, 1:2].flatten()
    
    scatter = ax.scatter(x, y, c=labels, s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)
    plt.savefig('out/viz_{}_{}'.format(k, image_name))

def clustering(image, num_clusters, image_name):
    image_shape = image.shape
    image_as_points = image.reshape(image_shape[0] * image_shape[1], 3)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(image_as_points)
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(image_as_points)
    new_image = centers[predict]
    new_image = new_image.reshape(image_shape)

    #sihlhouette = silhouette_score(image_as_points, predict)
    #print("clustering with {} clusters and silhouette {}".format(k, sihlhouette))
    
    plot_cluster(image_as_points, centers, predict, k, image_name)
    return new_image


image_dir = args.image
image_name = image_dir.split('/')[-1]

image = load_image(image_dir)

visualize_image(image, image_name)

for k in [2]:
    new_image = clustering(image, k, image_name)
    save_image('out/{}_{}.png'.format(image_name.split(".")[0], k), new_image)


