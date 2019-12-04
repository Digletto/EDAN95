import numpy as np
import operator


def gaussian_probability(x, mean, stdev):
    if stdev == 0 and x == mean:
        return 1
    elif stdev == 0 and x != mean:
        return 0
    else:
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent


def class_probability(label, value, class_meanstd, meanY, stdY):
    # not the actual probability calculation but maximises the same
    # needs division by overall value probability to be correct but
    # since that's the same division over all classes it doesn't matter for the argmax comparison
    probs = list()
    for i in range(len(value)):
        probs.append(gaussian_probability(value[i], class_meanstd[i][0], class_meanstd[i][1]))
    p_class = gaussian_probability(label, meanY, stdY)

    return np.prod(probs) * p_class


class GaussianNaiveBayes():
    def fit(self, x, y):
        divided = dict()
        for i in range(len(x)):
            label = y[i]
            value = x[i]
            if label not in divided:
                divided[label] = list()
            divided[label].append(value)

        self.mean = np.mean(x)
        self.std = np.std(x)
        self.meanY = np.mean(y)
        self.stdY = np.std(y)

        self.meanstd = dict()
        for label in divided:
            meanstd_list = list()
            data_matrix = np.vstack(divided[label])
            for column in data_matrix.T:
                meanstd_list.append((np.mean(column), np.std(column)))

            self.meanstd[label] = meanstd_list
        # print(self.meanstd[label])

    def all_class_probs(self, value):
        class_probs = dict()
        for label in self.meanstd:
            class_probs[label] = class_probability(label, value, self.meanstd[label], self.meanY, self.stdY)

        return class_probs

    def predict(self, x):
        # print('\033[91mNBC not implemented\033[0m')
        # print(x)

        predictions = list()
        for value in x:
            # print(value)
            probs = self.all_class_probs(value)
            predictions.append(max(probs.items(), key=operator.itemgetter(1))[0])

        return predictions
