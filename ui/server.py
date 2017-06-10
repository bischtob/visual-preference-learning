from flask import Flask, request
from flask import render_template, redirect
import numpy as np
from scipy.spatial import KDTree
import GPyOpt

STATIC_DIR = 'static'

app = Flask(__name__)

#================================================ Globals

# these will become part of a database.

data = {'imroot':'static/data/all/',
        'fpaths':None,
        'nn_tree':None,
        'kmeans':None,
        'images':None}

user = {'means':None,
        'counts':None,
        'scores':None,
        'score_prediction_error':None,
        'seen_images':set(),
        'seen':None,
        'index':None,
        'newest_image':None}
        

#================================================ Helper functions

def get_next_image():
    """
    Assumes newest_image was not None
    """
    global data, user

    # newest_image is a vector
    # dumb method: use NN tree to locate path
    dat,ind = data['nn_tree'].query(user['newest_image'], k=1)
    ind = ind[0]
    fpath = data['fpaths'][ind]

    return localize_image_path(fpath)


def localize_image_path(fpath):
    return data['imroot']+fpath.split('/')[-1][:-4] + ".jpg"


def sample_user_taste():
    """
    Use GPy's estimation at this time step
    to provide a best-guess estimate for the user's taste.
    (we may not be able to do this per-iteration, but only once at the end.)
    """
    global data, user

    # ah, right...is this possible?

    return []

#    (X_subset, Y_subset) = add_suggested_sample(user['myProblem'], 


def update_user_taste(score):
    """
    Update Gpy object with the new (image,score) pair
    """
    global data, user

    # print current score and prediction
    print("user score is: " + str(score))
    if user["score_prediction_error"] is not None:
        print("user score prediction error is: " + str(user["score_prediction_error"]))

    # update user's seen images
    if user['seen'] is None:
        user['seen'] = user['newest_image']
    else:
        user['seen'] = np.vstack((user['seen'], user['newest_image']))

 
    # add previous score to user's scores
    if user['scores'] is None:
        user['scores'] = np.atleast_2d(score)
    else:
        user['scores'] = np.vstack((user['scores'], score))

    # determine which type of step
    is_rand = False
    if user['seen'].shape[0]<3:
        is_rand = True
    else:
        (is_rand, user['temp']) = is_random_step(user['temp'])
    print("evaluation step is random: " + str(is_rand))

    # follow that step
    if is_rand:
        get_random_image()

    else:
        # run GPyOpt again
        domain = [{'name':'whocares', 'type':'bandit', 'domain':data['images']}]

        myProblem = GPyOpt.methods.BayesianOptimization(f = None,
                                                        X = user['seen'],
                                                        Y = user['scores'],
                                                        normalize_Y = False,
                                                        domain=domain)

        # get next suggested sample
        # (so we don't need to persist the GPyOpt problem)

        user['newest_image'] = myProblem.suggested_sample
        
        # add score prediction to user object
        mean_prediction, std_prediction = myProblem.model.predict(myProblem.suggested_sample)
        if user['score_prediction_error'] is None:
            user['score_prediction_error'] = np.atleast_2d(score-mean_prediction)
        else:
            user['score_prediction_error'] = np.vstack([user['score_prediction_error'], score-mean_prediction])

def is_random_step(t, c=0.65):
    """
    t temperature
    c adjustment to temp on random step
    returns (true/false, new_t)
    """
    if np.random.rand() < np.exp(-t):
        return (False, t)
    else:
        return (True, t*c)

def get_random_image():
    global data, user
    # get a random image to start
    ri = np.random.randint(0, len(data['images']))

    # kind of dumb since we're making get next image
    # do unnecessary calculation...oh well
    user['newest_image'] = np.atleast_2d(data['images'][ri])
 
def initialize_user():
    """
    Get a random initial image
    """
    global data, user

    get_random_image()

    user['temp'] = 2.0


def initialize_data():
    """
    Initialize the CNN embedding data and KDTree (for 
    computing nearest-neighbors)
    """
    global data, user

    # initialize CNN embedding data
    v = np.load('static/cnn_embedding.npz')
    data['fpaths'] = v['fpaths']
    data['images'] = v['emb']

    # initialize Nearest-Neighbors tree
    data['nn_tree'] = KDTree(data['images'])


#================================================ Flask logic

@app.before_first_request
def init():
    initialize_data()
    initialize_user()

@app.route('/', methods= ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', 
                image=get_next_image(),
                recs=sample_user_taste())

    if request.method == 'POST':
        # update user center estimate
        score = 5-float(request.form['button_press'])
        gsn_noise = np.random.normal(scale=0.2)
        update_user_taste(score+gsn_noise)

        # javascript will handle the refresh
        # 204 = no content
        return ('', 204)
