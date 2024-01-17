from sklearn import decomposition
def reduce_dim_fit(X):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X) # use a set of vectors to learn the PCA transformation
    return pca
def reduce_dim_infer(pca,X):
    Z = pca.transform(X) # transform a set of vectors to reduce their dim
    return Z # (it is possible that Z=X)