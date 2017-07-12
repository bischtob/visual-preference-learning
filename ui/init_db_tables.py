from server import db, Images, NNTree

import os
import numpy as np
from scipy.spatial import KDTree


def init_db_and_images():
    db.create_all()

    imroot = 'static/data/all/'
    v = np.load('static/cnn_embedding.npz')
    tff = v['fpaths']
    tif = v['emb']
    
    for i in range(len(tff)):
        fixed_im_path = imroot+(tff[i].split('/')[-1])[:-4]+'.jpg'
        db.session.add(Images(fixed_im_path, tif[i]))
    
    db.session.commit()


def init_nn_tree():
    """
    Initialize the KDTree (for computing nearest-neighbors)
    """
    # initialize Nearest-Neighbors tree
    # Images.query.all() is bad for RAM. this works more like an iterable
    # like range vs. xrange
    coords = np.array([img.coord for img in Images.query.yield_per(5).enable_eagerloads(False)])

    nn_tree = KDTree(coords)

    nt = NNTree(nn_tree)

    db.session.add(nt)
    db.session.commit()

# how to create one table
NNTree.__table__.create(db.session.bind)

init_nn_tree()
