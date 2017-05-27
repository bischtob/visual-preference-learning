from flask import Flask, request
from flask import render_template, redirect
import numpy as np
from scipy.spatial import KDTree

STATIC_DIR = 'static'

app = Flask(__name__)

#================================================ Globals

# these will become part of a database.

data = {'imroot':'static/data/all/',
        'fpaths':None,
        'embed':None,
        'nn_tree':None,
        'kmeans':None,
        'images':None}

user = {'means':None,
        'counts':None,
        'scores':None,
        'seen_images':set(),
        'last_image_seen':None,
        'index':None,
        'last_cluster_sampled':None}
        

#================================================ Helper functions

def get_next_image():
    global data, user

    index = user['index']
    # 0 to reference image path and not cluster id
    return data['images'][index][0]    
    
def fix_fpath(fpath):
    return data['imroot']+fpath.split('/')[-1]

def sample_user_taste():
    """
    out: list of paths to images representing
         the user's preferences.
    """
    global data, user

    # find the max-scored cluster so far
    max_ci = np.argmax(user['scores'])

    # if max is 0, return nothing
    if user['scores'][max_ci] == 0:
        return []
    
    mean_vector = data['kmeans'][max_ci]

    return sample_from_cluster(mean_vector, 5)


def sample_from_cluster(mean_vector, num):
    global data, user

    dist, ind = data['nn_tree'].query(mean_vector, k=num)

    return [fix_fpath(data['fpaths'][i]) for i in ind]


def update_user_taste(score):
    global data, user

    # reference the cluster id part
    # not sure why this needs to be int
    ci = int(data['images'][user['index']][1])

    user['scores'][ci] += score

    user['index'] += 1


def initialize_user_taste():
    global data, user

    user['scores'] = np.zeros(len(data['kmeans']))
    user['index'] = 0


def initialize_data():
    global data, user

    v = np.load('static/cnn_embedding.npz')
    data['fpaths'] = v['fpaths']
    data['embed'] = v['emb']

    # load the kmeans clusters
    temp = np.load('static/kmeans_clusters.npz')
    data['kmeans'] = temp['kmeans']

    # initialize nn tree (for recommendations)
    data['nn_tree'] = KDTree(data['embed'])

    # initialize images we will show by cluster
    imlist = []
    for i, cluster in enumerate(data['kmeans'][:100]):
        cid = [i]*3
        imlist += zip(sample_from_cluster(cluster, 3), cid)
    
    data['images'] = np.random.permutation(imlist)

    print data['images']
    
    # sanity check
    print 'total images: {0}'.format(len(imlist))


#================================================ Flask logic

@app.before_first_request
def init():
    initialize_data()
    initialize_user_taste()

@app.route('/', methods= ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', 
                image=get_next_image(),
                recs=sample_user_taste())

    if request.method == 'POST':
        # update user center estimate
        score = float(request.form['button_press'])
        update_user_taste(score)

        # javascript will handle the refresh
        # 204 = no content
        return ('', 204)


