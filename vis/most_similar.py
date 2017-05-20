import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

# load t-SNE
saved = np.load('emb_array.npz')
fpaths = saved['fpaths']
tsne_emb = saved['emb']

#tsne_emb = tsne_emb[:1000,:]

# run k-means
kmeans = KMeans(n_clusters=300, random_state=0).fit(tsne_emb)
print 'fit kmeans'

np.savez('kmeans_clusters', kmeans=kmeans.cluster_centers_)

tree = KDTree(tsne_emb)
print 'fit NN tree'

# get cluster centers


def plot_nearest(pt, tree):
    dist, ind = tree.query([pt], k=50)
    
    indices = ind[0][:]
    
    print indices
    
    image_tile = []
    
    for i in range(5):
        k = i*10
        image_tile.append([Image.open(fpaths[j]) for j in indices[k:k+10]])
    
    
    im_size = image_tile[0][0].size
    total_width,total_height = (len(image_tile)*im_size[0],len(image_tile[0])*im_size[1])
    
    new_im = Image.new('RGB', (total_width, total_height))
    
    x_offset = 0
    y_offset = 0
    for col in image_tile:
        for row in col:
            new_im.paste(row, (x_offset, y_offset))
            y_offset += im_size[1]
        y_offset = 0
        x_offset += im_size[0]
    
    new_im.show()

for c in kmeans.cluster_centers_:
    plot_nearest(c, tree)

# this was for one-dimensional

#im_size = image_tile[0].size
#
#total_width = im_size[0]*10
#total_height = im_size[1]
#
#new_im = Image.new('RGB', (total_width, total_height))

#x_offset = 0
#y_offset = 0
#for pic in image_tile:
#    new_im.paste(pic, (x_offset, y_offset))
#    x_offset += im_size[0]
#
#new_im.show()

