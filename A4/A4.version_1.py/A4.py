import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as plt
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from utilities.losses import compute_loss
from utilities.visualization import visualize_test
from utilities.visualization import visualize_train
from utilities.losses import compute_loss
from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn import preprocessing
import tensorflow as tf
# from utilities.optimizers import
seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3  # split the test set and traning set (3:7)
# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations


def onload():
    df = pd.read_csv("diamonds.csv", sep=',')
    print(df.shape)
    print(df.columns)
    print("data loaded")
    return df


def dataUnderstanding(df):
    """
    :type df: object
    """
    print("Data Head")
    print(df.head())
    print("Data information")
    print(df.info())
    print("Missing values")
    print(df.isnull().values.any())
    print(df.describe())
    histplot_data=df.drop('Count',axis=1)
    histplot_data.hist(bins=15, figsize=(14, 10))  # hisogram of all the data
    plt.show()
    print("Color distribution")
    print(df.color.value_counts())
    print("cut Distribution")
    print(df.cut.value_counts())
    #print("Clarity Distribution")
    #print(df.clarity.values_counts())

def data_preprocess(data):
    """
    :rtype: splitted training and test set
    """
    data_copy= data.copy()
    x_axis = data["carat"]
    y_axis = data["price"]
    plt.figure("2")
    plt.title("Carat Vs Actual Price")
    plt.scatter(x_axis,y_axis)
    plt.show()
    data_fixing = data[["cut","color","clarity"]]
    prepro = preprocessing.LabelEncoder()
    #convert all the data into integer
    data_fixing["cut"] = prepro.fit_transform(data_fixing["cut"])
    data_fixing["color"] = prepro.fit_transform(data_fixing["color"])
    data_fixing["clarity"] = prepro.fit_transform(data_fixing["clarity"])
    data_copy["cut"] = data_fixing["cut"]
    data_copy["color"] = data_fixing["color"]
    data_copy["clarity"] = data_fixing["clarity"]
    print("facorised data")
    print(data_copy.head())
    #print(data_copy.color.values_count())
    #print("check clarity")
    #print(data_copy["clarity"])
    #print(data_copy.clarity.values_counts())



    y_train = data["price"]
    train_data, test_data = train_test_split(data_copy,test_size=train_test_split_test_size)  # split the data into train and test set
    # pre-process data(both train and test)
    # remove the class label for data preprocessing and then we will use the full data for training and test
    train_data_full = train_data.copy()
    train_data = train_data.drop(["price"], axis=1)
    train_labels = train_data_full["price"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["price"], axis=1)
    test_labels = test_data_full["price"]
    train_data['intercept_dummy'] = pd.Series(1.0, index=train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index=test_data.index)
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


# def boxplots(data):
#    botplot1 = boxplots(data)


def learn(y, x, theta, max_iters, alpha, optimizer_type = "BGD", metric_type = "MSE"):
    thetas = None
    losses = None
    if optimizer_type == "BGD":
        thetas, losses = gradient_descent(y, x, theta, max_iters, alpha, metric_type)
    elif optimizer_type == "MiniBGD":
        thetas, losses = mini_batch_gradient_descent(y, x, theta, max_iters, alpha, metric_type, mini_batch_size = 10)
    elif optimizer_type == "PSO":
        thetas, losses = pso(y, x, theta, max_iters, 100, metric_type)
    else:
        raise ValueError(
            "[ERROR] The optimizer '{ot}' is not defined, please double check and re-run your program.".format(
                ot = optimizer_type))
    return thetas, losses


df = onload()
dataUnderstanding(df)
data_preprocess(df)