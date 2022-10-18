import numpy as np
import matplotlib.pyplot as plt
import models

NUM_OF_ITERATIONS = 500
HARD_C = 1e10


def draw_points(m):
    """
    Draw points from N(0, I_2) distribution and arrange them as data set of
    two, X and y
    :param m: An integer represents the number of samples needed to be drawn
    :return: A pair X; y where X is 2 x m matrix where each column represents
             an i.i.d sample from N(0, I_2) distribution, and y is its
             corresponding label, according to f(x)
    """
    mean = [0, 0]
    cov = np.identity(2)
    w = np.array([0.3, -0.5])

    legal_flag = True
    while legal_flag:

        X = np.random.multivariate_normal(mean, cov, m)

        y = np.sign(w.dot(X.T) + 0.1)

        # If we draw a training set where no point has y_i = 1 or no point has
        # y_i = -1 then just draw a new dataset instead, until we get points
        # from both types
        if np.sum(y) != m and np.sum(y) != -m:
            legal_flag = False

    return X, y


def classify_points(X, y):
    """
    Classifying data set according to its labels.
    :param X: Data set as matrix where each column represents an i.i.d samples
    :param y: The corresponding label
    :return: Four array's including the points from the original data set
             where any label and its corresponding data arranged
    """
    positive_x = []
    positive_y = []
    negative_x = []
    negative_y = []

    for point in range(len(y)):
        if y[point] == 1:
            positive_x.append(X[point][0])
            positive_y.append(X[point][1])
        else:
            negative_x.append(X[point][0])
            negative_y.append(X[point][1])

    return positive_x, positive_y, negative_x, negative_y


def q_9(arr):
    """
    For each m in the given array, draw m training points and plot figure as
    mentioned in the exercise instructions.
    :param arr: An array of integers represent count of training points to do
                the process on
    """
    for m in arr:
        X, y = draw_points(m)
        positive_x, positive_y, negative_x, negative_y = classify_points(X, y)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(positive_x, positive_y, '.', color='blue')
        ax.plot(negative_x, negative_y, '.', color='orange')

        x_true = np.linspace(np.min(X), np.max(X))
        w_true = [0.3, -0.5]
        a_true = -w_true[0] / w_true[1]
        y_true = a_true * x_true - 0.1 / w_true[1]
        ax.plot(x_true, y_true, color='tomato', label='True hyperplane')

        perceptron = models.Perceptron()
        x_perceptron = np.linspace(np.min(X), np.max(X))
        perceptron.fit(X.T, y)
        intercept = perceptron.model[0]
        w_perceptron = perceptron.model
        a_perceptron = -w_perceptron[1] / w_perceptron[2]
        y_perceptron = a_perceptron * x_perceptron - (intercept /
                                                      w_perceptron[2])
        ax.plot(x_perceptron, y_perceptron, color='darkblue',
                label='Perceptron hyperplane')

        svm = models.SVM(HARD_C)
        x_svm = np.linspace(np.min(X), np.max(X))
        svm.fit(X, y)
        a_svm = -svm.model[0] / svm.model[1]
        y_svm = a_svm * x_svm - svm.intercept / svm.model[1]
        ax.plot(x_svm, y_svm, color='green', label='SVM hyperplane')

        plt.legend()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.suptitle('Comparing hyperplanes of ' + str(m) + ' training points',
                     y=1, x=0.54)
        plt.title('from two-dimensional Gaussian distribution')

        plt.show()


def q_10(arr, k):
    """
    For each m in the given array, repeat the procedure mentioned in the
    exercise instructions 500 times with k = 10000 and save the accuracies of
    each classifier. Than, plot figure of the mean accuracy as function of m
    for each of the algorithms.
    :param arr: An array of integers represent count of training points to do
                the process on
    :param k: An integer represent the number of test points from N(0, I_2)
              distribution to calculate their labels
    """
    perceptron_acc_m = []
    svm_acc_m = []
    lda_acc_m = []

    for m in arr:
        perceptron_acc_itr = []
        svm_acc_itr = []
        lda_acc_itr = []

        for iteration in range(NUM_OF_ITERATIONS):
            X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(k)

            perceptron = models.Perceptron()
            perceptron.fit(X_train.T, y_train)
            perceptron.predict(X_test.T)
            perceptron.score(X_test, y_test)
            perceptron_acc_itr.append(perceptron.accuracy)

            svm = models.SVM(HARD_C)
            svm.fit(X_train, y_train)
            svm.predict(X_test.T)
            svm.score(X_test, y_test)
            svm_acc_itr.append(svm.accuracy)

            lda = models.LDA()
            lda.fit(X_train.T, y_train)
            lda.predict(X_test.T)
            lda.score(X_test, y_test)
            lda_acc_itr.append(lda.accuracy)

        perceptron_acc_m.append(np.mean(perceptron_acc_itr))
        svm_acc_m.append(np.mean(svm_acc_itr))
        lda_acc_m.append(np.mean(lda_acc_itr))

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(arr, perceptron_acc_m, label='Perceptron mean accuracy')
    ax.plot(arr, svm_acc_m, label='SVM mean accuracy')
    ax.plot(arr, lda_acc_m, label='LDA mean accuracy')

    plt.legend()

    plt.xlabel('m')
    plt.ylabel('Mean accuracy rate')
    plt.suptitle('Comparing the mean accuracy as function of m generated '
                 'by', y=1, x=0.54)
    plt.title('Perceptron, SVM and LDA models')

    plt.show()


def main():
    """
    Main function of the program, will call the functions represent the
    questions from the exercise.
    """
    arr = [5, 10, 15, 25, 70]
    k = 10000

    q_9(arr)
    q_10(arr, k)


if __name__ == '__main__':
    main()
