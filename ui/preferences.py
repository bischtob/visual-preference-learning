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

user_scores = []
last_image_shown = None
user_center_estimate = None
nn_tree = None

#================================================ Helper functions

def fix_fpath(fpath):
    return imroot+fpath.split('/')[-1]

def get_next_image():
    global last_image_shown
    # get a random image
    randint = np.random.randint(0, len(fpaths))

    # we need this to get the center
    last_image_shown = randint

    next_image = fix_fpath(fpaths[randint])

    return next_image

def get_recs():
    if user_center_estimate is None:
        return []
    else:
        print user_center_estimate
        dist, ind = nn_tree.query([user_center_estimate], k=6)
        # throw out the first result
        print ind
        print fpaths[ind[0][1]]
        return [fix_fpath(fpaths[i]) for i in ind[0][1:]]

def update_user_center_estimate(score):
    global user_scores, user_center_estimate

    # highest score is 4 (good), lowest is 0 (bad)
    # super hacky TODO
    if score > 0:
        score = 1.0/(5-score)
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


