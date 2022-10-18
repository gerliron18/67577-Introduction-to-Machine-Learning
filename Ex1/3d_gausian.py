import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
import math

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    """
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples
                  (first dimension for x, y, z coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def plot_2d(x_y):
    """
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples
                  (first dimension for x, y coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x_y[0]], [x_y[1]], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()


def q_11():
    """
    Use the identity matrix as the covariance matrix to
    generate random points and than plot them
    """
    q_11_x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(q_11_x_y_z)


S = np.matrix([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])


def q_12():
    """
    Transform the data with given scaling matrix and than plot the new points
    """
    q_12_x_y_z = S*np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(q_12_x_y_z)


rand_matrix = np.matrix(get_orthogonal_matrix(3))


def q_13():
    """
    Multiply the scaled data by random orthogonal matrix and than plot to new
    points
    """
    q_13_x_y_z = rand_matrix * S * np.random.multivariate_normal(mean, cov,
                                                                 50000).T
    plot_3d(q_13_x_y_z)


projection_matrix = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])


def q_14():
    """
    Plot the projection of the data to the x, y axes
    :return:
    """
    q_14_x_y_z = projection_matrix * rand_matrix * S * \
                 np.random.multivariate_normal(mean, cov, 50000).T
    plot_2d(q_14_x_y_z)


def q_15():
    """
    Plot the projection of the points to the x, y axes only for points
    where 0.1 > z > -0.4
    """
    q_15_x_y_z = projection_matrix * rand_matrix * S * \
                 np.random.multivariate_normal(mean, cov, 50000).T

    after_calc = q_15_x_y_z[0:2, np.where((x_y_z[2] > -0.4) &
                                          (x_y_z[2] < 0.1))]
    plot_2d(after_calc)


def find_mean(data, len, row):
    """
    Finds the mean of given array of data of length len*row
    :param data: given array of data
    :param len: the number of columns
    :param row: the number of rows
    :return: the mean of all data
    """
    sum = 0

    for i in range(len):
        sum += data[row][i]

    return sum / len


def q_16_a(data):
    """
    Plot the mean estimator as a function of number of samples in our dataset
    for the first 5 rows in the given data
    :param data: the given data array
    """
    x = []
    y = []

    for i in range(1, 1001):
        x.append(i)

    for j in range(1, 6):
        for k in range(1, 1001):
            y.append(find_mean(data,k,j))
        plt.plot(x, y)
        y = []

    plt.legend(("row_1", "row_2", "row_3", "row_4", "row_5"))
    plt.xlabel('Num of tosses')
    plt.ylabel('Estimated mean as function of m')

    plt.title("The mean estimator as a function of number of samples\nFor the "
              "first 5 sequences of 1000 tosses")
    plt.show()


epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]


def chebyshev(epsilon):
    """
    Find Chebyshev's bound of given epsilon
    :param epsilon: the given epsilon
    :return: array with Chebyshev's bounds for all 1000 and given epsilon
    """
    x = []
    y = []

    for i in range(1, 1001):
        x.append(i)

    for j in range(1, 1001):
        y.append(1 / (j * 4 * epsilon**2))

    return y


def hoeffding(epsilon):
    """
    Find Hoffding's bound of given epsilon
    :param epsilon: the given epsilon
    :return: array with Hoffding's bounds for all 1000 and given epsilon
    """
    x = []
    y = []

    for i in range(1, 1001):
        x.append(i)

    for j in range(1, 1001):
        y.append(2 * math.exp(-2 * j * epsilon**2))

    return x, y


def q_16_b():
    """
    plot the upper bound on Px1;...;xm(estimated_mean - E(X) >= epsilon)
    as a function of number of samples (which ranges from 1 to 1000)
    for each bound type and each epsilon
    """
    for i in epsilon:
        y_cheb = chebyshev(i)
        x, y_hoff = hoeffding(i)

        y_cheb = np.clip(y_cheb, 0, 1)
        y_hoff = np.clip(y_hoff, 0, 1)

        plt.plot(x, y_cheb)
        plt.plot(x, y_hoff)

        plt.xlabel('Num of tosses')
        plt.ylabel('Estimated mean as function of m')

        plt.title("Upper bound as function of m\nEpsilon = " + str(i))
        plt.legend(("chebyshev's inequality", "hoeffding's inequality"))

        plt.show()


def find_percentage(data, epsilon):
    """
    Find the percentage of sequences that satisfy
    (estimated_mean - E(X) >= epsilon) as a function of number of samples
    where E(x)=0.25 and for given epsilon
    :param data: the given data array
    :param epsilon: the given epsilon
    :return: (x, y) that satisfied (estimated_mean - E(X) >= epsilon)
    """
    x_sat = []
    y_sat = []

    for i in range(1, 1001):
        x_sat.append(i)

    num = np.arange(1, 1001)
    mean_estimator = np.cumsum(data, axis=1) / num
    mean_difference = np.abs(mean_estimator - 0.25)
    mean_accurate = np.where(mean_difference >= epsilon, 1, 0)
    res = np.sum(mean_accurate, axis=0)

    for j in range(1000):
        y_sat.append(res[j] / 100000)

    return x_sat, y_sat


def q_16_c(data):
    """
    Plot the percentage of sequences that satisfy
    (estimated_mean - E(X) >= epsilon) as a function of number of samples
    where E(x)=0.25 on top of the previous question plots
    :param data: the given data array
    """
    for i in epsilon:
        y_cheb = chebyshev(i)
        x, y_hoff = hoeffding(i)

        y_cheb = np.clip(y_cheb, 0, 1)
        y_hoff = np.clip(y_hoff, 0, 1)
        x_sat, y_sat = find_percentage(data, i)

        plt.plot(x, y_cheb)
        plt.plot(x, y_hoff)
        plt.plot(x_sat, y_sat)

        plt.xlabel('Num of tosses')
        plt.ylabel('Estimated mean as function of m')

        plt.title("The percentage of sequences satisfied with E(x)=0.25 as " +
                  "function of m\nEpsilon = " + str(i))
        plt.legend(("chebyshev's bound", "hoeffding's bound",
                    "Satisfied with E(x) = 0.25"))

        plt.show()


def main():
    """
    Main function of the program, will execute all practical exercises one
    by one
    """
    q_11()
    q_12()
    q_13()
    q_14()
    q_15()

    data = np.random.binomial(1, 0.25, (100000, 1000))
    q_16_a(data)
    q_16_b()
    q_16_c(data)


if __name__ == '__main__':
    main()
