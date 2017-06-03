"""
Ideas:
    
    Use to see where our model is failing (in this case, looks like "old
        people" feature prominently)
    
    Use as a measure of goodness of fit (if max outlier distance isn't
        very big, then maybe we can say the clustering is doing okay?)
    
    w/r/t the last idea: use as a way of hyperparameter tuning. plot
        median distance from clusters, or something, as a function of k,
        from small k to large k.

"""
import numpy as np
import os
from scipy.spatial import KDTree
import operator
from PIL import Image

fn = 'saved_kmeans_clusters/kmeans_clusters.npz'
kmeans = np.load(fn)['kmeans']

fn = '../ui/static/cnn_embedding.npz'
dat = np.load(fn)
(fpaths, coordinates) = (dat['fpaths'], dat['emb'])

tree = KDTree(kmeans)

def _score_func(coordinate, tree):
    # find nearest mean
    dist, ind = tree.query(coordinate, k=1)
    return dist

def detect_outliers(kmeans, coordinates):
    """
    kmeans - (k, dim) shape
    coords - (m, dim) shape

    This function detects outliers by sorting the coords according to
    the score function that assigns score d to a coord c, where d is
    dist(c, nearest mean)
    """

    scores = [(i, _score_func(c, tree)) for i, c in
             enumerate(coordinates)]

    return sorted(scores, key=operator.itemgetter(1), reverse=True)

def _localize_image_path(fpath):
    return os.path.join('../ui/static/data/all/', 
                        fpath.split('/')[-1])

# small scale test
num = 0
for im,score in detect_outliers(kmeans, coordinates)[:100]:
    print score
    fpath = _localize_image_path(fpaths[im])
    im = Image.open(fpath)
    im.save('outliers/out_'+str(num).zfill(3)+'.png', 'PNG')
    num += 1

