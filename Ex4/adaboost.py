"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg & Liron Gershuny
Date: May, 2020

"""
import numpy as np
import ex4_tools as tools
import matplotlib.pyplot as plt


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last
        iteration.
        """
        D = np.tile((1 / y.size), y.size)

        for i in range(self.T):
            h = self.WL(D, X, y)
            predicted = h.predict(X)
            miss = np.where(predicted != y)
            epsilon = np.sum(D[miss])

            w = 0.5 * np.log((1 / epsilon) - 1)

            D = D * np.exp(-y * w * predicted)

            D = D / np.sum(D)

            self.h[i] = h
            self.w[i] = w

        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        :param X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for
                      the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
                         Predict only with max_t weak learners
        """
        h_t = self.h[:max_t]
        sum = 0

        for i in range(len(h_t)):
            sum += self.w[i] * self.h[i].predict(X)

        y_hat = np.sign(sum)
        return y_hat

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        :param X : samples, shape=(num_samples, num_features)
        :param y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for
                      the classification
        :return: error : the ratio of the correct predictions when predict only
                         with max_t weak learners (float)
        """
        predicted = self.predict(X, max_t)
        miss = np.sum(predicted != y)
        error = miss / y.size

        return error


def q_10(T, noise_ratio=0):
    """
    Generate 5000 samples without noise, train an Adaboost classifier over this
    data and T = 500. Generate another 200 samples without noise ("test set")
    and plot the training error and test error, as a function of T.
    :param T: The number of base learners to learn
    :param noise_ratio: The noise rate of the data
    """
    train_X, train_y = tools.generate_data(5000, noise_ratio)
    test_X, test_y = tools.generate_data(200, noise_ratio)
    ada = AdaBoost(tools.DecisionStump, T)
    ada.train(train_X, train_y)

    train_err = []
    test_err = []
    for i in range(T):
        train_err.append(ada.error(train_X, train_y, i))
        test_err.append(ada.error(test_X, test_y, i))

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(np.arange(T), train_err, label='Train set')
    ax.plot(np.arange(T), test_err, label='Test set')

    plt.legend()

    plt.xlabel('T')
    plt.ylabel('Error rate')
    plt.title('The training error and test error as a function of T')

    plt.show()


def q_11(noise_ratio=0):
    """
    Plot the decisions of the learned classifiers with changing T's together
    with the test data.
    :param noise_ratio: The noise rate of the data
    """
    train_X, train_y = tools.generate_data(5000, noise_ratio)
    test_X, test_y = tools.generate_data(200, noise_ratio)

    T = [5, 10, 50, 100, 200, 500]
    count = 1
    for i in T:
        ada = AdaBoost(tools.DecisionStump, i)
        ada.train(train_X, train_y)
        plt.subplot(2, 3, count)
        tools.decision_boundaries(ada, test_X, test_y, i)
        count += 1

    plt.show()


def q_12(noise_ratio=0):
    """
    find T_hat, the one that minimizes the test error and plot the decision
    boundaries of this classifier together with the training data.
    :param noise_ratio: The noise rate of the data
    """
    train_X, train_y = tools.generate_data(5000, noise_ratio)
    test_X, test_y = tools.generate_data(200, noise_ratio)

    T = [5, 10, 50, 100, 200, 500]
    test_err = []
    hypothesis = []
    for i in T:
        ada = AdaBoost(tools.DecisionStump, i)
        ada.train(train_X, train_y)
        hypothesis.append(ada)
        test_err.append(ada.error(test_X, test_y, i))

    min_index = test_err.index(min(test_err))
    plt.suptitle('The test error is %s' % test_err[min_index], y=1)

    ada = hypothesis[min_index]
    tools.decision_boundaries(ada, train_X, train_y, T[min_index])
    plt.show()


def q_13(T, noise_ratio=0):
    """
    Take the weights of the samples in the last iteration of the training (D^T)
    Plot the training set with size proportional to its weight in D^T after
    normalizing them.
    :param T: The number of base learners to learn
    :param noise_ratio: The noise rate of the data
    """
    train_X, train_y = tools.generate_data(5000, noise_ratio)
    ada = AdaBoost(tools.DecisionStump, T)
    D = ada.train(train_X, train_y)
    D = D / np.max(D) * 10

    tools.decision_boundaries(ada, train_X, train_y, T, D)
    plt.suptitle('The training set with size proportional to its weight in D^T'
                 , y=1)
    plt.show()


def q_14(noise_arr):
    """
    Repeat q10-13 with noised data.
    :param noise_arr: An array of noise rates of the data
    """
    for i in noise_arr:
        q_10(500, i)
        q_11(i)
        q_12(i)
        q_13(500, i)


def main():
    """
    Main function of the program, will call other functions according to
    exercise instructions.
    """
    q_10(500)
    q_11()
    q_12()
    q_13(500)
    q_14([0.01, 0.4])


if __name__ == '__main__':
    main()
