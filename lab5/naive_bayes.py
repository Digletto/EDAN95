import numpy as np
import operator


def class_probability(label, value, divided, train, class_prob):
    # not the actual probability calculation but maximises the same
    # needs division by overall value probability to be correct but
    # since that's the same division over all classes it doesn't matter for the argmax comparison
    probs = list()
    divided_matrix = np.stack(divided[label])
    train_matrix = np.stack(train)
    for i in range(len(value)):
        occurences_total = np.sum((train_matrix[:, i] == value[i]))
        occurences_in_label = np.sum((divided_matrix[:, i] == value[i]))
        if occurences_total == 0:
            probs.append(0.0)
        else:
            probs.append(occurences_in_label/occurences_total)
    p_class = class_prob[label]

    return np.prod(probs) * p_class


class NaiveBayes:
    def fit(self, x, y):
        self.class_prob = dict()
        self.train = x
        self.divided = dict()
        for i in range(len(x)):
            label = y[i]
            value = x[i]
            if label not in self.divided:
                self.divided[label] = list()
            self.divided[label].append(value)

        for label in self.divided:
            self.class_prob[label] = y.tolist().count(label)/len(y)

    def all_class_probs(self, values):
        class_probs = dict()
        for label in self.divided:
            class_probs[label] = class_probability(label, values, self.divided, self.train, self.class_prob)

        return class_probs

    def predict(self, x):
        print('\033[91mNBC badly implemented\033[0m')
        predictions = list()
        for value in x:
            probs = self.all_class_probs(value)
            predictions.append(max(probs.items(), key=operator.itemgetter(1))[0])

        return predictions
