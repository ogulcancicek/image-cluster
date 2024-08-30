import numpy as np
from sklearn.decomposition import PCA


class DimReducer:
    def __init__(self, max_num_of_clusters, threshold=0.8):
        self.treshold = threshold
        self.max_num_of_clusters = max_num_of_clusters
        self.pca = None

    def fit(self, n_components, features):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(features)

    def transform(self, features):
        return self.pca.transform(features)

    def get_optimal_num_components(self, features):
        explained_variance = []

        for n_components in range(1, self.max_num_of_clusters, 50):
            pca = PCA(n_components)
            pca.fit(features)
            explained_variance.append(np.sum(pca.explained_variance_ratio_))

        optimal_num_comp = next(
            x for x, val in enumerate(explained_variance) if val >= self.treshold
        )

        return range(0, self.max_num_of_clusters, 50)[optimal_num_comp]
