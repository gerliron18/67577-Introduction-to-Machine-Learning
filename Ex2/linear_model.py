import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

FEATURES = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15',
            'sqft_lot15']


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


def predict(X, w):
    """
    :param X: A design matrix-numpy array with p rows and m columns
    :param w: Coefficients vector
    :return: A numpy array with the predicted value by the model.
    """
    predicted_values = np.matmul(X.T, w)

    return predicted_values


def mse(y, y_hat):
    """
    :param y: A response vector-numpy array
    :param y_hat: A prediction vector-numpy array
    :return: The MSE over the received samples
    """
    power_vector = np.power((y_hat - y), 2)
    final_mse = np.mean(power_vector)

    return final_mse


def load_data(path):
    """
    Given a path to the csv file, the function loads the dataset and performs
    all the needed preprocessing to get a valid design matrix.
    :param path: A path to the csv file
    :return: The dataset after the preprocessing.
    """
    raw_data = pd.read_csv(path, index_col=False)
    raw_data.fillna(-1, inplace=True)
    raw_data.drop(columns=['id', 'date', 'lat', 'long'], inplace=True)

    raw_data.drop(raw_data[raw_data['bedrooms'] < 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['bathrooms'] < 0.5].index, inplace=True)
    raw_data.drop(raw_data[raw_data['sqft_living'] <= 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['sqft_lot'] <= 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['floors'] < 1].index, inplace=True)
    raw_data.drop(raw_data[raw_data['waterfront'] < 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['view'] < 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['view'] > 4].index, inplace=True)
    raw_data.drop(raw_data[raw_data['condition'] < 1].index, inplace=True)
    raw_data.drop(raw_data[raw_data['condition'] > 5].index, inplace=True)
    raw_data.drop(raw_data[raw_data['grade'] < 1].index, inplace=True)
    raw_data.drop(raw_data[raw_data['grade'] > 13].index, inplace=True)
    raw_data.drop(raw_data[raw_data['sqft_above'] < 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['sqft_basement'] < 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['yr_built'] < 1900].index, inplace=True)
    raw_data.drop(raw_data[raw_data['yr_built'] > 2020].index, inplace=True)
    raw_data.drop(raw_data[raw_data['yr_renovated'] > 2020].index,
                  inplace=True)
    raw_data.drop(raw_data[raw_data['sqft_living15'] < 0].index, inplace=True)
    raw_data.drop(raw_data[raw_data['sqft_lot15'] < 0].index, inplace=True)

    raw_data.drop(raw_data[raw_data['price'] < 0].index, inplace=True)
    response_vector = raw_data['price']
    raw_data.drop(columns='price', inplace=True)

    zipcode_dum = pd.get_dummies(raw_data['zipcode'])
    raw_data.drop(columns=['zipcode'], inplace=True)
    ordered_data = raw_data.join(zipcode_dum)

    ordered_data.reset_index(drop= True, inplace=True)

    return ordered_data.T, response_vector


def plot_singular_values(data):
    """
    Plot a collection of singular values in descending order.
    That is: x-axis a running index number and
    y-axis the singular value's value
    :param data: A collection of singular values
    """
    sorted_data = np.sort(data)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(sorted_data[::-1], '*')
    ax.plot(sorted_data[::-1])

    plt.xlabel('Running index number')
    plt.ylabel('The singular values value')

    plt.title("Scree plot - the singular values in descending order")

    plt.show()


def q_15():
    """
    Loads the dataset, performs the preprocessing and plot
    the singular values plot.
    """
    X, y = load_data("kc_house_data.csv")
    w, s = fit_linear_regression(X, y)
    plot_singular_values(s)


def training(train_X, test_X, train_y, test_y):
    """
    Following q_16 function, over the 3/4 of the data for every p between 1
    and 100 fit a model based on the first p% of the training set.
    Then using the `predict` function test the performance of the fitted model
    on the test-set. Plots the results
    :param train_X: 3/4 of the data used as train-set
    :param test_X: 1/4 of the data used as test-set
    :param train_y: 3/4 of the response data used as train-response
    :param test_y: 1/4 of the response data used as test-response
    """
    mse_arr = []

    for p in range(1, 101):
        partition = math.ceil((p*train_X.shape[1])/100)
        part_x = train_X[:, :partition]
        part_y = train_y[:partition]
        w, s = fit_linear_regression(part_x, part_y)
        mse_train = mse(predict(test_X, w), test_y)
        mse_arr.append(mse_train)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(mse_arr)
    plt.xlabel('p% of the data')
    plt.ylabel('The MSE value')

    plt.title("The MSE over the test set as a function of p%")

    plt.show()


def q_16():
    """
    Fit a model and test it over the data. Splits the data into train and test
    sets randomly, such that the size of the test set is 1/4 of the total data.
    Next, will call the training function to plot the results.
    """
    X, y = load_data("kc_house_data.csv")

    X = X.to_numpy()
    y = y.to_numpy()

    train_X, test_X, train_y, test_y = train_test_split(X.T, y, test_size=0.25,
                                                        train_size=0.75)

    training(train_X.T, test_X.T, train_y, test_y)


def feature_evaluation(X, y):
    """
    Plot for every non-categorical feature, a graph (scatter plot) of the
    feature values and the response values. It then also computes and shows on
    the graph the Pearson Correlation between the feature and the response.
    :param X: A design matrix-numpy array
    :param y: A response vector-numpy array
    """
    for feature in FEATURES:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(X.T[feature], y, '.')

        ax.set_xlim(min(X.T[feature]), max(X.T[feature]))

        pearson_correlation = np.cov(X.T[feature], y) / (np.std(X.T[feature])
                                                         * np.std(y))

        plt.xlabel(feature + ' feature rate')
        plt.ylabel('Price rate')

        plt.title("The " + feature + " feature as function of the house " +
                  "price\nThe Pearson correlation is: " +
                  str(pearson_correlation[0][1]))

        plt.show()


def main():
    """
    Main function of the program, will execute question 15 and 16 and than
    load's the data and execute feature evaluation function
    """
    q_15()

    q_16()

    X, y = load_data("kc_house_data.csv")
    feature_evaluation(X, y)


if __name__ == '__main__':
    main()
