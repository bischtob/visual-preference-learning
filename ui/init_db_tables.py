from server import db, Images

import os
import numpy as np


def init_db_and_images():
    db.create_all()

    imroot = 'static/data/all/'
    v = np.load('static/cnn_embedding_compressed_p50.npz')
#    v = np.load('static/cnn_embedding.npz')
    tff = v['fpaths']
    tif = v['emb']
    
    for i in range(len(tff)):
        fixed_im_path = imroot+(tff[i].split('/')[-1])[:-4]+'.jpg'
        db.session.add(Images(fixed_im_path, tif[i]))
    
    db.session.commit()

init_db_and_images()
