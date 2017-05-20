import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

# load t-SNE
saved = np.load('emb_array.npz')
fpaths = saved['fpaths']
cnn_emb = saved['emb']

#tsne_emb = tsne_emb[:1000,:]

num = 300

# run k-means
kmeans = KMeans(n_clusters=num, random_state=0).fit(cnn_emb)
print 'fit kmeans'

np.savez('kmeans_clusters_'+str(num), kmeans=kmeans.cluster_centers_)

