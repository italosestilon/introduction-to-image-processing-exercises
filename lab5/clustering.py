
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('image', help="Image1's directory")
args = parser.parse_args()

if(not os.path.exists('out')):
    os.makedirs('out')

def load_image(dir):
    image = io.imread(dir)
    return img_as_float(image)

def save_image(dir, image):
    io.imsave(dir, img_as_ubyte(image))

def visualize_image(image_as_points, image_shape, image_name):
    pca = PCA(n_components=2)
    pca.fit(image_as_points)

    image_as_points_reduced = pca.transform(image_as_points)
   
    x = image_as_points_reduced[:, 0:1].flatten()
    y = image_as_points_reduced[:, 1:2].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=10, c='r', marker='o')
    plt.savefig('out/viz_{}'.format(image_name))

    return image_as_points_reduced

def plot_cluster(X, centers, labels, k, image_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = X[:,0:1].flatten()
    y = X[:, 1:2].flatten()
    
    scatter = ax.scatter(x, y, c=labels, s=10, cmap='tab20')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)
    plt.savefig('out/viz_{}_{}'.format(k, image_name))

def decode_image(codebook_dir, image_dir, image_shape):
    print("Decoding image")
    codebook = np.load(codebook_dir, allow_pickle=True)
    enc_image = np.load(image_dir, allow_pickle=True)

    image = codebook[enc_image]
    image = image.reshape(image_shape)

    return image

def clustering(image_as_points, image_shape, num_clusters, image_name, image_as_points_reduced):
    print("Clusteing image {} in {} clusters".format(image_name, k))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(image_as_points)
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(image_as_points)
    new_image = centers[predict]
    new_image = new_image.reshape(image_shape)

    print("Saving codebook")
    np.save('out/{}_{}.codebook.npy'.format(k, image_name.split('.')[0]), centers)
    print("Saving encoded image")
    np.save('out/{}_{}.enc.npy'.format(k, image_name.split('.')[0]), predict)

    sihlhouette = silhouette_score(image_as_points, predict, n_jobs=-1)

    with open('out/{}_silhouette.txt'.format(image_name.split('.')[0]), 'a') as f:
        f.write("clustering with {} clusters and silhouette {}\n".format(k, sihlhouette))
        
    print("clustering with {} clusters and silhouette {}".format(k, sihlhouette))
    
    plot_cluster(image_as_points_reduced, centers, predict, k, image_name)


image_dir = args.image
image_name = image_dir.split('/')[-1]

image = load_image(image_dir)

image_shape = image.shape
image_as_points = image.reshape(image_shape[0] * image_shape[1], 3)

image_as_points_reduced = visualize_image(image_as_points, image_shape, image_name)

for k in [2, 8, 16, 32, 64, 128]:
    clustering(image_as_points, image_shape, k, image_name, image_as_points_reduced)
    image_shape = image.shape
    codebook_dir = "out/{}_{}.codebook.npy".format(k, image_name.split('.')[0])
    enc_image_dir = "out/{}_{}.enc.npy".format(k, image_name.split('.')[0])

    new_image = decode_image(codebook_dir, enc_image_dir, image_shape)
    save_image('out/{}_{}.png'.format(image_name.split(".")[0], k), new_image)


