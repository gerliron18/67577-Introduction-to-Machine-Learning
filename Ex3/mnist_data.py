import numpy as np
import mnist
import matplotlib.pyplot as plt
import models
import timeit

NUM_OF_ITERATIONS = 50
SOFT_C = 100


def q_12(x_train, y_train):
    """
    Plot 3 images of samples labeled with '0' and 3 images of samples labeled
    with '1'
    :param x_train: mnist database of training x's
    :param y_train: mnist database of training labels y's
    """
    zero_count = 0
    one_count = 0
    for i in range(len(y_train)):
        if y_train[i] == 0 and zero_count < 3:
            img = x_train[i].reshape((28, 28))
            plt.imshow(img, cmap="Greys")
            plt.show()
            zero_count += 1
            continue

        if y_train[i] == 1 and one_count < 3:
            img = x_train[i].reshape((28, 28))
            plt.imshow(img, cmap="Greys")
            plt.show()
            one_count += 1
            continue


def rearrange_data(X):
    """
    :param X: A data as a tensor of size m x 28 x 28
    :return: New matrix of size m x 784 with the same data as the given one
    """
    return np.reshape(X, (X.shape[0], 784))


def draw_points(m, X, y):
    """
    Randomly draw m points from [0, len(X)]
    :param m: An integer represents the number of samples needed to be drawn
    :param X: Data set we need to draw points from
    :param y: The corresponding labels of X data
    :return: A pair X; y where X is a matrix where each column represents
             an i.i.d samples, and y is its corresponding labels
    """
    index = np.random.randint(0, len(X), m)

    X = X[index]
    y = y[index]

    return X, y


def q_14(arr, k, x_train_a, y_train_f, x_test_a, y_test_f):
    """
    For each m in the given array, repeat the procedure mentioned in the
    exercise instructions 50 times with k = 10000 and save the accuracies of
    each classifier. Than, plot figure of the mean accuracy as function of m
    for each of the algorithms.
    :param arr: An array of integers represent count of training points to do
                the process on
    :param k: An integer represent the number of test points to calculate their
              labels
    :param x_train_a: The train data set
    :param y_train_f: The train data labels
    :param x_test_a: The test data set
    :param y_test_f: The test data labels
    """
    logistic_acc_m = []
    svm_acc_m = []
    tree_acc_m = []
    nearest_neighbor_acc_m = []

    for m in arr:
        logistic_acc_itr = []
        svm_acc_itr = []
        tree_acc_itr = []
        nearest_neighbor_acc_itr = []

        # start = timeit.default_timer()
        for iteration in range(NUM_OF_ITERATIONS):
            X_train, y_train = draw_points(m, x_train_a, y_train_f)
            X_test, y_test = draw_points(k, x_test_a, y_test_f)

            logistic = models.Logistic()
            logistic.fit(X_train.T, y_train)
            logistic.predict(X_test.T)
            logistic.score(X_test, y_test)
            logistic_acc_itr.append(logistic.accuracy)


            svm = models.SVM(SOFT_C)
            svm.fit(X_train, y_train)
            svm.predict(X_test.T)
            svm.score(X_test, y_test)
            svm_acc_itr.append(svm.accuracy)

            tree = models.DecisionTree()
            tree.fit(X_train.T, y_train)
            tree.predict(X_test.T)
            tree.score(X_test, y_test)
            tree_acc_itr.append(tree.accuracy)

            nearest_neighbor = models.KNearestNeighbors()
            nearest_neighbor.fit(X_train.T, y_train)
            nearest_neighbor.predict(X_test.T)
            nearest_neighbor.score(X_test, y_test)
            nearest_neighbor_acc_itr.append(nearest_neighbor.accuracy)

        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

        logistic_acc_m.append(np.mean(logistic_acc_itr))
        svm_acc_m.append(np.mean(svm_acc_itr))
        tree_acc_m.append(np.mean(tree_acc_itr))
        nearest_neighbor_acc_m.append(np.mean(nearest_neighbor_acc_itr))

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(arr, logistic_acc_m, label='Logistic mean accuracy')
    ax.plot(arr, svm_acc_m, label='SVM mean accuracy')
    ax.plot(arr, tree_acc_m, label='Decision-Tree mean accuracy')
    ax.plot(arr, nearest_neighbor_acc_m, label='Nearest-Neighbor mean '
                                               'accuracy')

    plt.legend()

    plt.xlabel('m')
    plt.ylabel('Mean accuracy rate')
    plt.suptitle('Comparing the mean accuracy as function of m generated '
                 'by', y=1, x=0.54)
    plt.title('Logistic, SVM, Decision-Tree and k-Nearest-Neighbors models')

    plt.show()


def filter_data(X, y):
    """
    Filter the data set and its labels to 0, 1 numbers only
    :param X: Data set we need to filter
    :param y: The corresponding labels of X data needed to be filtered
    :return: A pair of X; y with data of numbers 0, 1 only
    """
    zero_index = np.where(y == 0)
    one_index = np.where(y == 1)

    all_indexes = np.concatenate((zero_index[0], one_index[0]))

    return X[all_indexes], y[all_indexes].astype(np.int)


def main():
    """
    Main function of the program, will call the functions represent the
    questions from the exercise.
    """
    x_train_all, y_train_all = mnist.train_images(), mnist.train_labels()
    x_test_all, y_test_all = mnist.test_images(), mnist.test_labels()

    x_train_f, y_train_f = filter_data(x_train_all, y_train_all)
    x_test_f, y_test_f = filter_data(x_test_all, y_test_all)

    q_12(x_train_f, y_train_f)

    arr = [50, 100, 300, 500]
    k = 1000
    x_train_a = rearrange_data(x_train_f)
    x_test_a = rearrange_data(x_test_f)
    y_train_f[y_train_f == 0] = -1
    y_test_f[y_test_f == 0] = -1

    q_14(arr, k, x_train_a, y_train_f, x_test_a, y_test_f)


if __name__ == '__main__':
    main()
