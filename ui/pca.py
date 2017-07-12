import numpy as np

from sklearn.decomposition import PCA

filename_in = "static/cnn_embedding.npz"
filename_out = "static/cnn_embedding_compressed_p50.npz"

data = np.load(filename_in)
fpaths = data['fpaths']
emb = data['emb']

pca = PCA(n_components=20)
pca.fit(emb)
emb_transformed = pca.transform(emb)
print(np.sum(pca.explained_variance_ratio_))

np.savez(filename_out, fpaths=fpaths, emb=emb)

