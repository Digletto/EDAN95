import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NearestCentroid():
    def fit(self, x, y):
        classes = np.unique(y)
        self.reverse = dict(enumerate(classes))
        self.encoding = {v: k for k, v in self.reverse.items()}

        self.centroids = np.zeros((len(classes), x.shape[1]))

        for label in classes:
            centroid = x[y == label].mean(axis=0)
            self.centroids[self.encoding[label]] = centroid

    def predict(self, x):
        class_indexes = euclidean_distances(x, self.centroids).argmin(axis=1)

        return np.array([self.reverse[i] for i in class_indexes])
