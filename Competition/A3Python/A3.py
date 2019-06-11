import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# reading the data set
def read_dataset():
         data= pd.read_csv('1.csv')
         print(len(data.columns))
         x = data[data.columns[0:58]].values
         y = data[data.columns[58]]
         # Encode the dependent variable
         encoder = sklearn.preprocessing.LabelEncoder()
         encoder.fit(y)
         y=encoder.transform(y)
         Y= sklearn.preprocessing.OneHotEncoder(y)
         print(x.shape)
         print(data.describe())
         print(data)
         return (x,Y,y,data)

def relace_missing_value():
        read_dataset()
