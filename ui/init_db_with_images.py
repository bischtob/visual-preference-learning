"""
Adds all the images to the database.
Should only need to be run once.
"""
from server import db, Images

import os
import numpy as np

db.create_all()

imroot = 'static/data/all/'
v = np.load('static/cnn_embedding.npz')
tff = v['fpaths']
tif = v['emb']

for i in range(len(tff)):
    fixed_im_path = imroot+tff[i].split('/')[-1]
    db.session.add(Images(fixed_im_path, tif[i]))

db.session.commit()
