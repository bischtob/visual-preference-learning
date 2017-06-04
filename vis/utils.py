import numpy as np
import os
from scipy.spatial import KDTree
import operator
from PIL import Image

def initial_loading():
    fn = 'saved_kmeans_clusters/kmeans_clusters.npz'
    kmeans = np.load(fn)['kmeans']
    
    fn = '../ui/static/cnn_embedding.npz'
    dat = np.load(fn)
    (fpaths, coordinates) = (dat['fpaths'], dat['emb'])
    
#    tree = KDTree(kmeans)

    return (fpaths, coordinates, kmeans)

def localize_image_path(fpath):
    return os.path.join('../ui/static/data/all/', 
                        fpath.split('/')[-1])

