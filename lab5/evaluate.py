import pandas as pd
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

import data
from gaussian_bayes import GaussianNaiveBayes as GNB
from naive_bayes import NaiveBayes as NBC
from nearest_centroid import NearestCentroid as NCC


def _evaluate_accuracy(clfs, x_train, x_test, y_train, y_test):
    res = []

    for clf in clfs:
        if not clf:
            res.append(None)
        else:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            res.append(metrics.accuracy_score(y_test, y_pred))

    return res


def evaluate():
    names = ['Sklearn NCC', 'Sklearn GNB', 'Custom NCC', 'Custom GNB', 'Custom NBC']

    results = {
        'Sklearn digits':
        _evaluate_accuracy([NearestCentroid(), GaussianNB(),
                            NCC(), GNB(), NBC()], *data.load_sk_digits()),
        'Sklearn digits summarized':
        _evaluate_accuracy([NearestCentroid(), GaussianNB(),
                            NCC(), GNB(), NBC()], *data.load_sk_digits_summarized()),
        'MNIST Light':
        _evaluate_accuracy(
            [NearestCentroid(), GaussianNB(), NCC(), GNB(), None], *data.load_light_digits()),
    }

    print(pd.DataFrame(results, index=names))


if __name__ == "__main__":
    evaluate()
