import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import randoma
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ridge_regression
from sklearn.metrics import mean_absolute_error
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
    print(df.head())
    print("data loaded")
    print("cut Distribution")
    print(df.cut.value_counts())
    print("Clarity Distribution")
    print(df.clarity.value_counts())
    print("Color distribution")
    print(df.color.value_counts())
    print("Missing values:", df.isnull().values.any())
    return df
def convert_categorical_value_to_numeric(df):
    df['cut'] = df['cut'].map({'Ideal':5, 'Premium':4, 'Very Good':3,'Good': 2,'Fair':1})
    df['clarity'] = df['clarity'].map({'IF':8,'VVS1':7,'VVS2':6,'VS1':5,'VS2':4,'SI1':3,'SI2':2,'I1':1})
    df['color'] =df['color'].map({'J':7, 'I': 6, 'H':5, 'G':4, 'F':3,'E':2,'D':1})
    df.drop(['Count'],axis=1)#drop the index number
    print(df.head())
    return df;

def split_test_train(df):
    train_data,test_data = train_test_split(df,test_size=0.3,random_state=seed)
    print("train data describe:")
    print(train_data.describe())
    print("train data correlation")
    print(train_data.corr())
    train_data_copy = train_data.copy()
    train_be4_standard = train_data_copy.drop(['price'],axis=1)
    test_be4_standard = test_data.copy()
    test_be4_standard = test_be4_standard.drop(['price'],axis=1)
    train_mean = train_be4_standard.mean()
    train_std = train_be4_standard.std()
    train_standarded = (train_be4_standard-train_mean) / train_std
    test_standarded = (test_be4_standard - train_mean) /train_std
    test_standarded['price'] = test_data['price']
    train_standarded['price'] = train_data['price']
    # Tricks: add dummy intercept to both train and test
    # train_standarded['intercept_dummy'] = pd.Series(1.0, index=train_standarded.index)
    # test_data['intercept_dummy'] = pd.Series(1.0, index=test_standarded.index)
    return train_standarded,test_standarded
def drop_class_label(test_df):
    # copy of the data
    test_data_copy=test_df.copy();
    test_without_label = test_data_copy.drop(["price"],axis=1)
    test_label = test_data_copy["price"]
    print("dropped label test data:")
    print(test_without_label.head())
    return test_without_label, test_label
#build the linear model
def build_model(train_data,technique):
    start_time = datetime.datetime.now()  # Track learning starting time
    if(technique=="Linear Regression"):
        baseline = LinearRegression()
    elif technique == "KNR":
        baseline = KNeighborsRegressor(n_neighbors=10)
    elif technique == "Ridge regression":
        from sklearn.linear_model import Ridge
        # train_data_copy = train_data.copy()
        # train_data_copy_nolabel = train_data_copy.drop(["price"], axis=1)
        # train_data_copy_label = train_data["price"]
        # baseline = Ridge()
        # feededmodel = baseline.fit(train_data_copy_nolabel, train_data_copy_label)
        baseline = Ridge()
    elif technique == "decision tree regression":
        from sklearn.tree import DecisionTreeRegressor
        baseline = DecisionTreeRegressor()
    elif technique == "random forest regression":
        from sklearn.ensemble import RandomForestRegressor
        baseline = RandomForestRegressor()
    elif technique == "gradient Boosting regression":
        from sklearn.ensemble import GradientBoostingRegressor
        baseline = GradientBoostingRegressor(max_depth=10,n_estimators=200)
    elif technique == "SGD regression":
        from sklearn.linear_model import SGDRegressor
        baseline = SGDRegressor()
    elif technique == "support vector regression":
        from sklearn.svm import SVR
        baseline = SVR()
    elif technique == "multi-layer perceptron regression":
        from sklearn.neural_network import MLPRegressor
        baseline = MLPRegressor(early_stopping=True,learning_rate_init=0.5)
    elif technique == "linear SVR":
        from sklearn.svm import LinearSVR
        baseline = LinearSVR()
    else:
        print("no such technique:" + technique)
        return 0;

    train_data_copy = train_data.copy()
    train_data_copy_nolabel = train_data_copy.drop(["price"],axis=1)
    train_data_copy_label = train_data["price"]
    feededmodel = baseline.fit(train_data_copy_nolabel,train_data_copy_label)
    print(feededmodel)
    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = (end_time - start_time).total_seconds()  # Track execution time
    return feededmodel,exection_time

def predict_test(test_without_label,model,test_label):
    y_pred = model.predict(test_without_label)
    # print("coefficient:", model.coef_)
    # print("Intercept:",model.intercept_)
    # print("model score:", model.score(test_without_label, test_label))
    mse= mean_squared_error(test_label,y_pred)
    # print("MSE: {error}".format(error=mse))
    r2_error = r2_score(test_label,y_pred)
    # print("R2: {error}:".format(error=r2_error))
    return test_without_label,test_label,y_pred,model

def score(test_result,testlabel,y_pred,model):
    print("model-score:", model.score(test_result,testlabel))
    mse = mean_squared_error(testlabel, y_pred)
    print("MSE: {error}".format(error=mse))
    print("RSME: {error}".format(error=np.sqrt(mse)))
    r2_error = r2_score(testlabel,y_pred)
    print("R2: {error}:".format(error=r2_error))
    mae = mean_absolute_error(testlabel,y_pred)
    print("Mean absolute: {error}".format(error=mae))



if __name__ == '__main__':
    df=onload()
    df_converted=convert_categorical_value_to_numeric(df)
    train_data,test_data=split_test_train(df_converted)
    test_without_label,test_label=drop_class_label(test_data)

    #modelarray = []
    #*************************************************************
    #**************************************************************
    # building model and testing start *******************
    print("*********************")
    print("Linear Regression:")
    technique = "Linear Regression"
    model,execution_time= build_model(train_data,technique)
    test_result,testlabel,y_pred,model= predict_test(test_without_label,model,test_label)
    score(test_result,testlabel,y_pred,model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("KNR:")
    technique = "KNR"
    model,execution_time = build_model(train_data,technique)
    test_result, testlabel, y_pred, model= predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)
    print("*********************")
    print("Ridge Regression:")
    technique = "Ridge regression"
    model,execution_time = build_model(train_data,technique)
    test_result, testlabel, y_pred, model= predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("decision tree regression")
    technique = "decision tree regression"
    model,execution_time= build_model(train_data,technique)
    test_result, testlabel, y_pred, model= predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("random forest regression")
    technique = "random forest regression"
    model,execution_time = build_model(train_data,technique)
    test_result, testlabel, y_pred, model= predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    #
    print("*********************")
    print("gradient Boosting regression")
    technique = "gradient Boosting regression"
    model,execution_time = build_model(train_data,technique)
    test_result, testlabel, y_pred, model= predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("SGD regression")
    technique = "SGD regression"
    model,execution_time = build_model(train_data, technique)
    test_result, testlabel, y_pred, model = predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("support vector regression")
    technique = "support vector regression"
    model,execution_time = build_model(train_data, technique)
    test_result, testlabel, y_pred, model = predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("linearSVR regression")
    technique = "linear SVR"
    model,execution_time = build_model(train_data, technique)
    test_result, testlabel, y_pred, model = predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)

    print("*********************")
    print("multi-layer perceptron regression")
    technique = "multi-layer perceptron regression"
    model,execution_time = build_model(train_data, technique)
    test_result, testlabel, y_pred, model = predict_test(test_without_label, model, test_label)
    score(test_result, testlabel, y_pred, model)
    print("execution time :")
    print(execution_time)