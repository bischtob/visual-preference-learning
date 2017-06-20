# web app imports
from flask import Flask, request, session
from flask import render_template, redirect
from flask_sqlalchemy import SQLAlchemy

# machine learning imports
import numpy as np
from scipy.spatial import KDTree
import GPyOpt

#================================================ Startup

STATIC_DIR = 'static'

app = Flask(__name__)

# TODO: something more secure...
app.config['SECRET_KEY'] = 'adsfsafsa'

#================================================ SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

# the user in the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    newest_image = db.Column(db.Integer)
    temp = db.Column(db.Float)

    def __init__(self, newest_image):
        self.newest_image = newest_image
        self.temp = 2.0


class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    image_id = db.Column(db.Integer)
    score = db.Column(db.Integer)

    def __init__(self, user_id, image_id, score):
        self.user_id = user_id
        self.image_id = image_id
        self.score = score

# this class should be populated offline and persisted
class Images(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fpath = db.Column(db.String)
    # 128-dim numpy array
    coord = db.Column(db.PickleType)

    def __init__(self, fpath, coord):
        self.fpath = fpath
        self.coord = coord

#================================================ Globals

nn_tree = None

#================================================ Helper functions

def get_img_from_scores(user_scores):
    """
    user_scores is a list of db objects
    we need the images to be in the same order as the scores
    """
    coords = []
    for score in user_scores:
        img = Images.query.get(score.image_id)
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
    user = get_current_user(session.get('user_id'))

    img = Images.query.get(user.newest_image)

    print img.fpath
    
    return img.fpath


def sample_user_taste():
    """
    TODO
    Use GPy's estimation at this time step
    to provide a best-guess estimate for the user's taste.
    (we may not be able to do this per-iteration, but only once at the end.)
    """

    return []

def get_current_user(uid):
    return User.query.get(uid)

def is_random_step(init_thresh = 3):
    """
    if user has seen fewer than 3 images, always random step.
    otherwise random step according to temperature.
    """
    user = get_current_user(session.get('user_id'))

    n_seen_img = len(Score.query.filter_by(user_id=user.id).all())

    print n_seen_img

    if n_seen_img < init_thresh:
        print 'hi' # should print x3
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

    Has no reference to a user.
    """
    coords = np.array([img.coord for img in Images.query.all()])


    # name doesn't matter
    domain = [{'name':'x', 'type':'bandit', 'domain':coords}]

    print x.shape
    print y.shape

    # f is not needed. Our problem is defined solely by X and Y.
    myProblem = GPyOpt.methods.BayesianOptimization(f = None, 
                                                    X = x,
                                                    Y = y,
                                                    normalize_Y = False,
                                                    domain=domain)

    # get next suggested sample here, so we don't have to persist GPyOpt object.
    gpy_next = myProblem.suggested_sample

    # use nearest-neighbor to get next sample.
    dat,ind = nn_tree.query(gpy_next, k=1)

    # convert ndarray with one entry to int
    return ind[0]


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
        print 'random step'
        user.newest_image = uniform_random_image()
    else: # gpyopt step
        print 'gpyopt step'

        # format for GPyOpt
        images = get_img_from_scores(user_scores)
        scores = reshape_user_scores(user_scores)

        # run gpyopt and get next suggestion
        user.newest_image = update_gpyopt(images, scores)


    print user.newest_image    
    # in all cases commit changes to database
    db.session.commit()
    

def initialize_user():
    """
    Get a random initial image, and set the user's temperature value.

    TODO: manage user in session

    """

    user = User(uniform_random_image())

    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id

    print 'user with id %d initialized' % session.get('user_id')

    # login user, then can use current_user to do things?


def uniform_random_image():
    """
    Return 'newest image' without gpyopt.
    Used for random annealing steps and for initialization.
    """
    n_imgs = db.session.query(Images).count()
    newest_image = np.random.randint(0, n_imgs)

    return newest_image


def initialize_data():
    """
    Initialize the KDTree (for computing nearest-neighbors)
    TODO: persist this in filesystem
    """

    global nn_tree

    # initialize Nearest-Neighbors tree
    coords = np.array([img.coord for img in Images.query.all()])
    nn_tree = KDTree(coords)


def consume_score(score):
    """
    Abstract these details away from the routing code.
    """
    score = 6-float(score)
    gsn_noise = np.random.normal(scale=0.2)
    update_user_taste(score+gsn_noise)


#================================================ Flask logic

@app.before_first_request
def init():
    initialize_data()
    initialize_user()

@app.route('/', methods= ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', 
                image=get_newest_image_path(),
                recs=sample_user_taste())

    if request.method == 'POST':
        # update user center estimate
        consume_score(request.form['button_press'])

        # javascript will handle the refresh
        # 204 = no content
        return ('', 204)

