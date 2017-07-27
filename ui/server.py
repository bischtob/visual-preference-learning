# misc imports
from __future__ import print_function
import os, sys

# web app imports
from flask import Flask, request, session
from flask import render_template, redirect
from flask_sqlalchemy import SQLAlchemy

# machine learning imports
import numpy as np
import GPyOpt

#================================================ Startup

STATIC_DIR = 'static'

app = Flask(__name__)

def read_secret_key(filename='secret_key'):
    """
    http://flask.pocoo.org/snippets/104/
    """

    filename = os.path.join(app.instance_path, filename)
    try:
        return open(filename, 'rb').read()
    except IOError:
        print('Error: No secret key.')
        sys.exit(1)


app.config['SECRET_KEY'] = read_secret_key()

#================================================ SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

class User(db.Model):
    """
    Database object to model the user.
    """
    id = db.Column(db.Integer, primary_key=True)
    newest_image = db.Column(db.Integer)
    temp = db.Column(db.Float)

    def __init__(self, newest_image):
        self.newest_image = newest_image
        self.temp = 2.0


class Score(db.Model):
    """
    For a given image and user, the score the user gave to the image.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    image_id = db.Column(db.Integer)
    score = db.Column(db.Integer)

    def __init__(self, user_id, image_id, score):
        self.user_id = user_id
        self.image_id = image_id
        self.score = score


class PredictedScore(db.Model):
    """
    For a given image and user, the score GPyOpt expected that user to give.
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    image_id = db.Column(db.Integer)
    score = db.Column(db.Integer)

    def __init__(self, user_id, image_id, score):
        self.user_id = user_id
        self.image_id = image_id
        self.score = score


class Images(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fpath = db.Column(db.String)
    # 128-dim numpy array
    coord = db.Column(db.PickleType)

    def __init__(self, fpath, coord):
        self.fpath = fpath
        self.coord = coord

#================================================ Helper functions

def get_img_from_scores(user_scores):
    """
    user_scores is a list of db objects
    we need the images to be in the same order as the scores
    """
    coords = []

    ids_seen = [] # debugging - this is correct.
    image_ids_seen = []
    for score in user_scores:
        ids_seen.append(score.image_id)

        # score.image_id is the correct id.
        # somehow the returned image is wrong.
        img = Images.query.get(score.image_id)

        image_ids_seen.append(img.id)

        coord = img.coord
        coords.append(coord)

    return np.array(coords)


def reshape_user_scores(user_scores):
    """
    Format user scores from database into a 2d numpy array so that 
    GPyOpt can consume it as a parameter.
    """
    return np.array([np.array([s.score]) for s in user_scores])

def get_newest_image_path():
    """
    Assumes newest_image is an id.
    """

    if 'user_id' not in session:
        print('new user initialized')
        initialize_user()

    user = get_current_user(session.get('user_id'))

    print(session.get('user_id'))

    img = Images.query.get(user.newest_image)

    return img.fpath


def sample_user_taste(nsamples = 25):
    """
    Use GPy's estimation at this time step
    to provide a best-guess estimate for the user's taste.

    Note: SLOW.
    """

    # if user does not exist, return random stuff
    if 'user_id' not in session:
        best = np.random.randint(0, len(coords), nsamples)
        worst = np.random.randint(0, len(coords), nsamples)
    else:
        user = get_current_user(session.get('user_id'))
    
        coords = np.array([img.coord for img in Images.query.all()])
        
        # retrieve the Score objects produced by this user
        user_scores = Score.query.filter_by(user_id=user.id).all()
    
        # if too few scores, return random stuff
        if len(user_scores) < 2:
            best = np.random.randint(0, len(coords), nsamples)
            worst = np.random.randint(0, len(coords), nsamples)
        else:
            # format for GPyOpt
            images = get_img_from_scores(user_scores)
            scores = reshape_user_scores(user_scores)
        
            # name doesn't matter
            domain = [{'name':'whocares', 'type':'bandit', 'domain':coords}]
        
            # f is not needed. Our problem is defined solely by X and Y.
            myProblem = GPyOpt.methods.BayesianOptimization(f = None,
                                                            X = images,
                                                            Y = scores,
                                                            normalize_Y = False,
                                                            domain=domain)
        
            # iterate through all samples to find top and bottom
            # TODO: naive brute-force method, can be improved
            
            # -1 as default index to cause error downstream if it's still around
            best = [(-1, float('-inf'))]*nsamples
            worst = [(-1, float('inf'))]*nsamples
            
            for i,coord in enumerate(coords):
                mean_prediction, std_prediction = myProblem.model.predict(coord)
                mp = mean_prediction[0][0] # mean prediction is 2d
            
                # remember that the problem is minimization? so best-worst 
                # are flipped.
                
                mp = 6 - mp
                
                # replace the lowest 'high' value if mp is higher
                (mini, minv) = _find_min(best)
                if mp > minv:
                    # we are adding a tuple of the image index, and its predicted score.
                    # so we can retrieve the image path later from the index
                    best[mini] = (i, mp)
                
                # replace the highest "low" value if mp is lower
                (maxi, maxv) = _find_max(worst)
                if mp < maxv:
                    worst[maxi] = (i, mp)

            # change format of best/worst
            best = [b[0] for b in best]
            worst = [w[0] for w in worst]
    
    # TODO: fill random?
    random = []
    best_img_paths = _get_image_paths_from_list(best)
    worst_img_paths = _get_image_paths_from_list(worst)
    
    return (best_img_paths, worst_img_paths, random)

def _get_image_paths_from_list(l):
    image_paths = []
    for i in l:
        # database is 1-indexed
        # also, make sure int is not a numpy int
        img = Images.query.get(int(i)+1)
        image_paths.append(img.fpath) 

    return image_paths
    
# naive: a min-heap would be better
def _find_min(best):
    mini = 0
    minv = best[0][1]
    for i,val in enumerate(best):
        if val[1] < minv:
            minv = val[1]
            mini = i

    return (mini, minv)


def _find_max(worst):
    maxi = 0
    maxv = worst[0][1]
    for i,val in enumerate(worst):
        if val[1] > maxv:
            maxv = val[1]
            maxi = i

    return (maxi, maxv)
 
def get_current_user(uid):
    return User.query.get(uid)


def is_random_step(init_thresh = 3):
    """
    if user has seen fewer than 3 images, always random step.
    otherwise random step according to temperature.
    """
    user = get_current_user(session.get('user_id'))

    n_seen_img = len(Score.query.filter_by(user_id=user.id).all())

    if n_seen_img < init_thresh:
        return True
    else:
        # does this modify the database entry?
        (is_rand, user.temp) = annealing_step(user.temp)

         # commit the temperature update
        db.session.commit()
    
        return is_rand


def annealing_step(t, c=0.65):
    """
    t temperature
    c adjustment to temp on random step
    returns (true/false, new_t)
    """
    if np.random.rand() < np.exp(-t):
        return (False, t)
    else:
        return (True, t*c)


def update_gpyopt(x, y):
    """
    Run GPyOpt with the current x and y. Return suggested_sample, 
    squeezed to one dimension. 

    Also create PredictedScore object for the suggested sample.
    """

    user = get_current_user(session.get('user_id'))

    coords = np.array([img.coord for img in Images.query.all()])
    
    # name doesn't matter
    domain = [{'name':'whocares', 'type':'bandit', 'domain':coords}]

    # f is not needed. Our problem is defined solely by X and Y.
    myProblem = GPyOpt.methods.BayesianOptimization(f = None, 
                                                    X = x,
                                                    Y = y,
                                                    normalize_Y = False,
                                                    domain=domain)

    # get next suggested sample here, so we don't have to persist GPyOpt object.
    gpy_next = myProblem.suggested_sample
    gpy_ind = myProblem.acquisition_optimizer.x_min_index

    # database is 1-indexed, so we need to add one.
    # numpy int64s behaving funky
    image_id = int(gpy_ind)+1
    print('image id from gpy field: %d' % image_id)
    
    # create PredictedScore
    mean_prediction, std_prediction = myProblem.model.predict(gpy_next)
    mp = mean_prediction[0][0] # mean prediction is 2d
    db.session.add(PredictedScore(user.id, image_id, mp))
    db.session.commit()

    print('next image: %d' % image_id)
    return image_id


def update_user_taste(score):
    """
    Update Gpy object with the new (image,score) pair
    """
    user = get_current_user(session.get('user_id'))

    # create a new "Score" entry

    db.session.add(Score(user_id=user.id, score=score, image_id=user.newest_image))
    db.session.commit()

    # retrieve the Score objects produced by this user
    user_scores = Score.query.filter_by(user_id=user.id).all()

    # user_scores needs to be appropriately formatted for GPyOpt

    # determine which type of step
    if is_random_step(): # random step
        print('random step')
        user.newest_image = uniform_random_image()
    else: # gpyopt step
        print('gpyopt step')

        # format for GPyOpt
        images = get_img_from_scores(user_scores)
        scores = reshape_user_scores(user_scores)

        # run gpyopt and get next suggestion
        user.newest_image = update_gpyopt(images, scores)

    # in all cases commit changes to database
    db.session.commit()
    

def initialize_user():
    """
    Initialize new user object with:
        Random initial image
        Temperature (for annealing, same initial val for all users)
    """

    user = User(uniform_random_image())

    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id

    print('user with id %d initialized' % session.get('user_id'))


def uniform_random_image():
    """
    Return 'newest image' without gpyopt.
    Used for random annealing steps and for initialization.
    """

    n_imgs = db.session.query(Images).count()
    newest_image = np.random.randint(0, n_imgs)

    # database is 1-indexed
    return newest_image+1


def consume_score(score):
    """
    Abstract these details away from the routing code.
    """
    score = 6-float(score)
    gsn_noise = np.random.normal(scale=0.2)
    update_user_taste(score+gsn_noise)


#================================================ Flask logic

@app.route('/', methods= ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', 
                image=get_newest_image_path())

    if request.method == 'POST':
        val = request.form['button_press']
        if val=='reset':
            initialize_user()
        else:
            # update user center estimate
            consume_score(val)
    
        # javascript will handle the refresh
        # 204 = no content
        return ('', 204)

@app.route('/results', methods= ['GET'])
def results():
    (top_imgs, bottom_imgs, random_imgs) = sample_user_taste()
    return render_template('results.html',
        top_images = top_imgs,
        bottom_images = bottom_imgs,
        random_images = random_imgs)

