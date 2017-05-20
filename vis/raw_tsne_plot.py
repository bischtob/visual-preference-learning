import numpy as np
from PIL import Image

import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png

def load_tsne(path):
    saved = np.load('tsne_array.npz')
    fpaths = saved['fpaths']
    tsne_emb = saved['tsne_emb'] # these will be our coordinates
    
    return fpaths,tsne_emb


# initialization
fig = PLT.gcf()
fig.clf()
ax = PLT.subplot(111)

# load t-SNE
fpaths,tsne_emb = load_tsne('tsne_emb.npz')


def add_image(path, xy):
    arr_hand = read_png(path)
    imagebox = OffsetImage(arr_hand, zoom=0.3)

    ab = AnnotationBbox(imagebox, xy,
        xycoords='data', frameon=False,
        boxcoords='offset points')

    ax.add_artist(ab)

#i = 0
#while i < 500:
#
#    randi = np.random.randint(0, len(tsne_emb))
#
#    add_image(fpaths[randi], tsne_emb[randi])
#    i+= 1

#X = tsne_emb[:,0]
#Y = tsne_emb[:,1]

X = []
Y = []

i = 0
while i < 1000:
    randi = np.random.randint(0, len(tsne_emb))
    X.append(tsne_emb[randi,0])
    Y.append(tsne_emb[randi,1])
    i += 1

#ax.grid(True)
#ax.set_ylim([-2, 2])
#ax.set_xlim([-2, 2])
#PLT.draw()

PLT.scatter(X, Y)
PLT.show()
