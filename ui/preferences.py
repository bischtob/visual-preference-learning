from flask import Flask, request
from flask import render_template, redirect
import numpy as np
from scipy.spatial import KDTree

STATIC_DIR = 'static'

app = Flask(__name__)

#================================================ Globals
imroot = 'static/data/all/'
fpaths = None
embed = None

seen_images = set()

user_scores = []
last_image_shown = None
user_center_estimate = None
nn_tree = None

#================================================ Helper functions

def fix_fpath(fpath):
    return imroot+fpath.split('/')[-1]

def get_next_image():
    global last_image_shown, seen_images

    selection = None

    if user_center_estimate is None:
        # get a random image
        selection = np.random.randint(0, len(fpaths))

    else:
        # get an image near cluster center
        dist, ind = nn_tree.query([user_center_estimate], k=100)
        ind = ind[0]

        # TODO: hack
        selection = ind[0]
        j = 0
        while selection in seen_images:
            j += 1
            selection = ind[j]

    # we need this to get the center
    last_image_shown = selection

    seen_images.add(last_image_shown)

    next_image = fix_fpath(fpaths[selection])

    return next_image

def get_recs():
    if user_center_estimate is None:
        return []
    else:
        dist, ind = nn_tree.query([user_center_estimate], k=6)
        # throw out the first result (duplicate)
        return [fix_fpath(fpaths[i]) for i in ind[0][1:]]

def update_user_center_estimate(score):
    global user_scores, user_center_estimate

    # highest score is 4 (good), lowest is 0 (bad)
    # super hacky TODO
    if score > 2:
        # 1 for 4, 1/2 for 3, 1/4 for 2
        score = 1.0/(5-score)
    else: # penalize (SUPER HACKY)
        # -1 for 0, -1/2 for 1
        score = -1.0/(1+score)
    user_scores.append(score*embed[last_image_shown,:])
    user_center_estimate = np.array(reduce(lambda x,y: x+y, user_scores))

#================================================ Flask logic

@app.before_first_request
def init():
    global fpaths, embed, nn_tree
    # load the npz array, grab an image at random
    dat = np.load('static/cnn_embedding.npz')
    fpaths = dat['fpaths']
    embed = dat['emb']

    # initialize nn tree (for recommendations)
    nn_tree = KDTree(embed)

@app.route('/', methods= ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', 
                image=get_next_image(),
                recs=get_recs())

    if request.method == 'POST':
        # update user center estimate
        score = float(request.form['button_press'])
        update_user_center_estimate(score)

        return redirect('/')


