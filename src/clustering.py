from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ImageClustering:
    def __init__(self, min_clusters=5, max_clusters=20):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.best_n_clusters = None
        self.best_labels_ = None
        self.best_score = -1

    def fit(self, features):
        best_score = -1
        best_labels_ = None
        best_n_clusters = self.min_clusters

        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            model = KMeans(n_clusters=n_clusters, random_state=0)
            labels = model.fit_predict(features)

            score = silhouette_score(features, labels)

            if score > best_score:
                best_score = score
                best_labels_ = labels
                best_n_clusters = n_clusters

        self.best_score = best_score
        self.best_labels_ = best_labels_
        self.best_n_clusters = best_n_clusters

    def predict(self, features):
        if self.best_labels_ is None:
            raise ValueError("You need to fit the model before predicting.")
        return self.best_labels_

    def get_best_score(self):
        if self.best_score == -1:
            raise ValueError("You need to fit the model before getting the best score.")
        return self.best_score

    def get_best_n_clusters(self):
        if self.best_n_clusters is None:
            raise ValueError(
                "You need to fit the model before getting the best number of clusters."
            )
        return self.best_n_clusters
