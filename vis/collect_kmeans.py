"""
Various clustering-related functionality.
(kmeans, tsne, vis, etc.)

collect_kmeans
visualize_centers
load_cnn_embedding
collect_tsne

Dev notes: possible refactoring is to move the KMeans and TSNE imports to 
the functions that require them.

"""

from __future__ import print_function
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

from utils import initial_loading, localize_image_path


def collect_kmeans(cnn_embedding, n_clusters, out_dir):
    """
    Loads the CNN embedding and runs k-means 
    on the embedded space.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cnn_embedding)
    print('Fit kmeans with {0} clusters.'.format(n_clusters))
    
    fn_out = os.path.join(out_dir, 'kmeans_clusters_'+str(n_clusters))
    np.savez(fn_out, kmeans=kmeans.cluster_centers_)

    print('Saved cluster centers to '+fn_out+'.npz')
    
    # return centers as well, in case you want to do something with them
    return kmeans.cluster_centers_

def _plot_nearest(pt, tree, tile_width):

    # find tile_width*tile_width nearest
    dist, ind = tree.query([pt], k=tile_width**2)
    
    indices = ind[0][:]
    
    image_tile = []
    
    # create tile_width x tile_width array
    for i in range(tile_width):
        k = i*tile_width
        image_tile.append([Image.open(localize_image_path(fpaths[j])) for j in indices[k:k+tile_width]])
    
    im_size = image_tile[0][0].size
    total_width,total_height = (len(image_tile)*im_size[0],len(image_tile[0])*im_size[1])
    
    new_im = Image.new('RGB', (total_width, total_height))
    
    x_offset = 0
    y_offset = 0
    for col in image_tile:
        for row in col:
            new_im.paste(row, (x_offset, y_offset))
            y_offset += im_size[1]
        y_offset = 0
        x_offset += im_size[0]
    
    return new_im


def visualize_centers(kmeans_centers, cnn_embedding, tile_width=3):
    """
    Given kmeans centers and the CNN embedding 
    coordinates, produces a visualization of 
    the cluster centers.

    Not really restricted to centers--
    any points will work.
    """

    # precomputation for nearest-neighbor
    tree = KDTree(cnn_embedding)

    # each call to plot_nearest does an image.show()
    # so you're getting kmeans_centers
    for i,c in enumerate(kmeans_centers):
        im = _plot_nearest(c, tree, tile_width)
        im.save('centers/'+str(i).zfill(3)+'.png', 'PNG')


def load_cnn_embedding(path_to_cnn_embedding):
    """
    Easier to have this in one location
    if it needs to be changed.
    """
    saved = np.load(path_to_cnn_embedding)
    
    return (saved['fpaths'], saved['emb'])


def collect_tsne(cnn_embedding, out_dir, n_components=2):
    """
    Computes T-SNE reduction on the CNN embeddings.
    Takes a few hours on my laptop.
    """
    model = TSNE(n_components=n_components, random_sate=0)

    # this call takes ~3 hours -- be patient!
    tsne_embedding = model.fit_transform(cnn_embedding)

    fn_out = os.path.join(out_dir, 'tsne_'+str(n_components))
    np.savez(fn_out, tsne=tsne_embedding)

    print('Saved T-SNE with {0} components to {1}'.format(n_components, fn_out))


def collect_pca(cnn_embedding, n_components=100):
    """
    We never used this, really. Saving just in case.
    """
    pca = PCA(n_components=n_components)
    pca.fit(cnn_embedding)
    print(pca.explained_variance_ratio_)


if __name__ == '__main__':
    (fpaths, coordinates, kmeans) = initial_loading()
    visualize_centers(kmeans, coordinates)
