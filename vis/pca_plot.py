import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

v = np.load('cnn_embedding.npz')
fpaths = v['fpaths']
emb = v['emb']

subset = emb[:2000,:]
print subset.shape

pca = PCA(n_components=100)
pca.fit(subset)
print(pca.explained_variance_ratio_)

#X = pca.fit_transform(subset)

#plt.scatter(X[:,0], X[:,1])
#plt.show()
