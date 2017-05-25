from flask import Flask, request
from flask import render_template, redirect
import numpy as np
from scipy.spatial import KDTree

STATIC_DIR = 'static'

app = Flask(__name__)

#================================================ Globals
seen_images = set()

scores = None
last_image_shown = None
user_center_estimate = None
nn_tree = None

user_means = None
user_counts = None

last_cluster = None

# these things will become part of a database.

data = {'imroot':'static/data/all/',
        'fpaths':None,
        'embed':None,
        'nn_tree':None }

user = {'means':None,
        'counts':None,
        'scores':None,
        'seen_images':set()}
        

#================================================ Helper functions

def sample_multinomial():
    # really inefficient
    ms = np.random.multinomial(1, normalize(user_counts))
    ms = np.nonzero(ms)
    # why is this necessary...?
    return ms[0][0]

def sample_uniform():
    # TODO supah hack
    ms = np.random.multinomial(1, [0.2, 0.2, 0.2, 0.2, 0.2])
    ms = np.nonzero(ms)
    print 'cluster #{0} was sampled'.format(ms[0][0])
    # why is this necessary...?
    return ms[0][0]

def fix_fpath(fpath):
    return imroot+fpath.split('/')[-1]

def get_next_image():
    global last_image_shown, seen_images, last_cluster, scores

    # sample from clusters evenly (exploration)
#    ms = sample_uniform()

    # sample for cluster with the worst score
    print scores
    ms = np.argmin(scores)

    # if score in this cluster is positive (exploitation)
    if scores[ms] > 0:
        dist, ind = nn_tree.query(user_means[ms,:], k=100)

        selection = ind[0]

        j = 0
        while selection in seen_images:
            j += 1
            selection = ind[j]

    # otherwise reset this cluster and random sample (exploration)
    else:
        scores[ms] = 0.0
        selection = np.random.randint(0, len(embed))
        user_means[ms] = embed[selection]
        user_counts[ms] = 0

    # we need this to get the center
    last_image_shown = selection
    last_cluster = ms

    seen_images.add(last_image_shown)

    next_image = fix_fpath(fpaths[selection])

    return next_image

def normalize(v):
    norm = np.sum(v)
    if norm==0:
        return v
    return v/norm

def get_recs():
    # sample with multinomial distribution according to counts...
    # five draws
    if np.sum(user_counts)<=10:
        return []

    mean_samples = [sample_multinomial() for i in range(5)]

    # for each sample, run nn query with that cluster's mean
    indices = []
    # one for each cluster
    for ms in range(5):

        dist, ind = nn_tree.query([user_means[ms,:]], k=1)
        indices.append(ind[0])

    return [fix_fpath(fpaths[i]) for i in indices]

def adjust_score(score):
    if score > 1:
        # 1 for 4, 1/2 for 3, 1/4 for 2
        score = 1.0/(5-score)
    else: # penalize, but less so (SUPER HACKY)
        # -1 for 0, -1/2 for 1
        score = -1.0/(3+score)

#    if score > 1:
#        score -= 1
#    else:
#        score = 0

    print 'score: {0}'.format(score)
    return score

def update_user_center_estimate(score):
    global user_means, user_counts, scores

    closest = last_cluster
    print 'updating cluster #{0}'.format(closest)

    # use the negative scores for this part
    # but not for other parts
    scores[closest] += adjust_score(score)

    user_counts[closest] += 1.0
    # modify to take score into account--
    # we want to push the mean away from a bad score
    # but maybe this updating is wrong...?
    update = adjust_score(score)*(1/user_counts[closest])*(embed[last_image_shown]-user_means[closest])
    

    print 'user counts'
    print user_counts

    user_means[closest] += update

def get_max_norm(v):
    max_norm = float("-inf")
    for e in v:
        npl = np.linalg.norm(e)
        if npl > max_norm:
            max_norm = npl

    return max_norm

def initialize_user_means(k):
    global user_means

    # initialize means by picking random samples
    randints = [np.random.randint(0, len(embed)) for i in range(k)]
    user_means = np.array([embed[i,:] for i in randints])

    # sanity check
    max_embed = get_max_norm(embed)
    max_user = get_max_norm(user_means)

    print 'max norm: {0}'.format(max_embed)
    print 'user means max norm: {0}'.format(max_user)

    # no need to return -- global

#================================================ Flask logic

@app.before_first_request
def init():
    # bad but we will fix
    global data, user 
    # load the npz array, grab an image at random
    v = np.load('static/cnn_embedding.npz')
    data['fpaths'] = v['fpaths']
    data['embed'] = v['emb']

    # initialize nn tree (for recommendations)
    nn_tree = KDTree(embed)

    # initialize random means; TODO hacky
    k = 5
    initialize_user_means(k)
    user_counts = np.zeros(k)
    scores = np.zeros(k)

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

        # javascript will handle it
        # 204 = no content
        return ('', 204)


