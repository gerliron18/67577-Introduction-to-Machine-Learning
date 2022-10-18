import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr


def fit_linear_regression(X, y):
    """
    :param X: A design matrix-numpy array with p rows and n columns
    :param y: A response vector-numpy array with n rows
    :return: Two sets of values: the first is a numpy array of the coefficients
             vector `w` and the second is a numpy array of the singular values
             of X.
    """
    x_inverse = np.linalg.pinv(X)
    w = np.matmul(x_inverse.T, y)

    singular_values = np.linalg.svd(X, full_matrices=False, compute_uv=False,
                                    hermitian=False)

    return w, singular_values


def main():
    """
    Main function of the program, will execute questions 18 to 21 a.k.a
    read the covid-19 data, preform preproccesing and than fit the data and
    plot two figures with the linear regression real values compared to the
    estimated values and the exponential regression real values compared to
    the estimated values.
    """

    # q_18
    covid_df = pd.read_csv("covid19_israel.csv", index_col=False)

    # q_19
    covid_df['log_detected'] = np.log(covid_df['detected'])

    # q_20
    X = np.array(covid_df['day_num']).reshape(len(covid_df['day_num']), 1)
    y = np.array(covid_df['log_detected']).reshape(
        len(covid_df['log_detected']), 1)

    w, s = fit_linear_regression(X.T, y)

    y_hat_linear = np.matmul(X, w)
    y_hat_log = np.exp(y_hat_linear)

    # q_21
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(covid_df['day_num'], covid_df['log_detected'], '*')
    ax.plot(covid_df['day_num'], y_hat_linear, color='red')
    plt.xlabel('Count of days since first infection was identified in Israel')
    plt.ylabel('Log of the sum of the number of detected cases')
    plt.legend(("Real data", "Estimated linear curve"))

    plt.title("The log of the detected cases as a function of day count")

    plt.show()

    fig2 = plt.figure()
    ax = fig2.add_subplot()
    plt.plot(covid_df['day_num'], covid_df['detected'], '*')
    ax.plot(covid_df['day_num'], y_hat_log, color='red')
    plt.xlabel('Count of days since first infection was identified in Israel')
    plt.ylabel('The number of detected cases')
    plt.legend(("Real data", "Estimated exponential curve"))

    plt.title("The detected cases as a function of day count")

    plt.show()


if __name__ == '__main__':
    main()
