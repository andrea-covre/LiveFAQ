import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def fit_clustering(X, n, method):
    model = None
    labels = None
    if method == "gmm-full":
        model = GaussianMixture(n_components=n, random_state=0, covariance_type="full").fit(X)
        labels = model.predict(X)
    elif method == "gmm-diag":
        model = GaussianMixture(n_components=n, random_state=0, covariance_type="diag").fit(X)
        labels = model.predict(X)
    elif method == "kmeans":
        print("h1")
        model =  KMeans(n_clusters=n, random_state=0).fit(X)
        print("h2")
        labels = model.labels_
        print("h3")
    else:
        print("Unrecognized method. Sticking with GMM with diagonal covariance")
        model = GaussianMixture(n_components=n, random_state=0, covariance_type="diag").fit(X)
        labels = model.predict(X)
    return model, labels