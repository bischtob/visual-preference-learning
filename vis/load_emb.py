# load embedding array

import numpy as np
import operator
from sklearn.manifold import TSNE
from PIL import Image

def vector_distance(x, y):
    return (x[0]-y[0])**2+(x[1]-y[1])**2

# will change filename
v = np.load('emb_array.npy.npz')
fpaths = v['fpaths']
cnn_emb = v['emb']

# run t-SNE on emb vectors
model = TSNE(n_components=2, random_state=0)
tsne_emb = model.fit_transform(cnn_emb)

# save t-SNE

# find closest vectors
min_dist = float("inf")
min_pair = None

for i,x in enumerate(cnn_emb):
    for j,y in enumerate(cnn_emb[i+1:]):
        vd = vector_distance(x,y)
        if vd<min_dist:
            min_dist = vd
            min_pair = [fpaths[i],fpaths[j]]

# sort in two dimensions.

# sort on x
aligned = []
for i,fp in enumerate(fpaths):
    aligned.append((fp, tsne_emb[i][0], tsne_emb[i][1]))

pregroup = sorted(aligned, key=operator.itemgetter(1), reverse=True)

# bucket the list sorted on x
buckets = []
for i in range(10):
    buckets.append(pregroup[10*i:10*i+10])

# now sort each bucket individually on y
sbuckets = [sorted(b, key=operator.itemgetter(2)) for b in buckets]

# create image tile

imagetile = []
for sbucket in sbuckets:
    column = map(Image.open, [dat[0] for dat in sbucket])
    imagetile.append(column)

# hardcoding -- assumes all images same size
im_size = imagetile[0][0].size
total_width,total_height = (10*im_size[0],10*im_size[1])

new_im = Image.new('RGB', (total_width, total_height))

x_offset = 0
y_offset = 0
for col in imagetile:
    for row in col:
        new_im.paste(row, (x_offset, y_offset))
        y_offset += im_size[1]
    y_offset = 0
    x_offset += im_size[0]

new_im.show()

# show images

#images = map(Image.open, min_pair)
#widths, heights = zip(*(i.size for i in images))
#
#total_width = sum(widths)
#max_height = max(heights)
#
#new_im = Image.new('RGB', (total_width, max_height))
#
#x_offset = 0
#for im in images:
#    new_im.paste(im, (x_offset,0))
#    x_offset += im.size[0]
#
#new_im.show()


# fill every position with nearest neighbor
