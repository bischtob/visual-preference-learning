import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree

# load t-SNE
saved = np.load('tsne_array.npz')
fpaths = saved['fpaths']
tsne_emb = saved['tsne_emb']

# find min/max y and x values.
# hacky way
min_y = float("inf")
max_y = float("-inf")
min_x = float("inf")
max_x = float("-inf")

for val in tsne_emb:
    if val[1] < min_y:
        min_y = val[1]
    elif val[1] > max_y:
        max_y = val[1]

    if val[0] < min_x:
        min_x = val[0]
    elif val[0] > max_x:
        max_x = val[0]


print min_y,max_y,min_x,max_x

min_y = np.floor(min_y)
max_y = np.ceil(max_y)
min_x = np.floor(min_x)
max_x = np.ceil(max_x)

# create a 100x100 image tile (or some value) numpy matrix where each element is a (x,y, path) tuple, where x and y are based on the min/max x/y values from the whole dataset.

tiledim = 50

step = (max_x-min_x)/float(tiledim)

# this is very useful.
X,Y = np.mgrid[min_x:max_x:step, min_y:max_y:step]
tile = np.dstack((X,Y))

# for each such pair, perform nearest neighbor search to find the closest vector, and assign path.
paths = np.empty((tile.shape[0],tile.shape[1]), dtype=object)

# gross
paths = [['']*tile.shape[1] for i in range(tile.shape[0])]

# create tree for NN queries
#tree = BallTree(tsne_emb)
tree = KDTree(tsne_emb)

already_used = set()

# this is stupid
for i in range(tile.shape[0]):
    for j in range(tile.shape[1]):
        val = tile[i,j] # this is [x,y]..right

        # run nearest-neighbor search
        dist, ind = tree.query([val], k=10)

        indsel = ind[0][0] # why?

        # temporarily turn off for high-dim
#
#        indi = 0
#
#        while indsel in already_used:
#            indsel = ind[0][indi]
#            indi += 1
#
#        print indsel
#        already_used.add(indsel)

        # assign fpath to paths
        paths[i][j] = fpaths[indsel]


# many of these are duplicates

# use existing code to: create new matrix that is matrix of images, and visualize this image.

image_tile = [[Image.open(fp) for fp in l] for l in paths]

im_size = image_tile[0][0].size
total_width,total_height = (tiledim*im_size[0],tiledim*im_size[1])

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

