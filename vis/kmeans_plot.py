import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

# load CNN embedding
saved = np.load('emb_array.npz')
fpaths = saved['fpaths']
tsne_emb = saved['emb']

# load existing k-means clusters
v = np.load('kmeans_clusters_300.npz')
kmeans_centers = v['kmeans']

# fit tree for nearest-neighbors
tree = KDTree(tsne_emb)


def plot_nearest(pt, tree):
    dist, ind = tree.query([pt], k=50)
    
    indices = ind[0][:]
    
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

for c in kmeans_centers[:5]:
    plot_nearest(c, tree)

