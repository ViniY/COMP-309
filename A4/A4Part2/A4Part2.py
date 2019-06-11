import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ridge_regression
import seaborn as sns
seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3  # split the test set and traning set (3:7)
# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations

def onload():
    train_df= pd.read_csv("adult.data", sep=',',header=None)
    test_df = pd.read_csv("adult.test", skiprows=[0],sep=',',header=None)#skip the first row
    train_df = train_df.replace(" ?",value= np.NaN)
    test_df = test_df.replace(" ?",value= np.NaN)
    print(train_df.shape)
    print(train_df.columns)
    print(train_df.head())
    print("Train Missing values:", train_df.isnull().values.any())
    print("Test Missing values:", test_df.isnull().values.any())
    array_head = train_df.head()
    sns.heatmap(train_df.isnull(), cbar=False)
    plt.figure()
    plt.plot
    plt.show()
    train_df = train_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    test_df = test_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    print("Train Missing values after replacing nan with most frequent:", train_df.isnull().values.any())
    print("Test Missing values:", test_df.isnull().values.any())
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df[1] = le.fit_transform(train_df[1])
    train_df[3] = le.fit_transform(train_df[3])
    train_df[5] = le.fit_transform(train_df[5])
    train_df[6] = le.fit_transform(train_df[6])
    train_df[7] = le.fit_transform(train_df[7])
    train_df[8] = le.fit_transform(train_df[8])
    train_df[9] = le.fit_transform(train_df[9])
    train_df[13] = le.fit_transform(train_df[13])
    train_df[14] = le.fit_transform(train_df[14])
    test_df[1] = le.fit_transform(test_df[1])
    test_df[3] = le.fit_transform(test_df[3])
    test_df[5] = le.fit_transform(test_df[5])
    test_df[6] = le.fit_transform(test_df[6])
    test_df[7] = le.fit_transform(test_df[7])
    test_df[8] = le.fit_transform(test_df[8])
    test_df[9] = le.fit_transform(test_df[9])
    test_df[13] = le.fit_transform(test_df[13])
    test_df[14] = le.fit_transform(test_df[14])
    return train_df,test_df

def build_model(train_df,train_label,technique):
    train_df_copy= train_df.copy()
    test_df_copy = test_df.copy()
    # train_df_nolabel = train_df_copy.drop(train_df_copy.columns[len(train_df_copy.columns)-1], axis=1)
    # print(train_df_nolabel.describe())
    train_df_nolabel = train_df.iloc[:, :-1]
    train_label = train_df.iloc[:,-1]
    train_df_label = train_df_copy.columns[len(train_df_copy.columns)-1]
    #********* KNN
    if technique == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier()#default k = 10
        model = neigh.fit(train_df_nolabel.values,train_label)
        return model
    elif technique == "Naive Bayes":
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        model = gnb.fit(train_df_nolabel.values,train_label)
    elif technique == "SVM":
        from sklearn.svm import SVC
        svm_model = SVC()
        model = svm_model.fit(train_df_nolabel.values,train_label)
    elif technique =="Decesion Tree":
        from sklearn.tree import DecisionTreeClassifier
        dT = DecisionTreeClassifier();
        model = dT.fit(train_df_nolabel.values,train_label)
    elif technique == "Random Forest" :
        from sklearn.ensemble import RandomForestClassifier
        Rf = RandomForestClassifier()
        model = Rf.fit(train_df_nolabel.values,train_label)
    elif technique == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        ada = AdaBoostClassifier()
        model = ada.fit(train_df_nolabel.values,train_label)
    elif technique == "Linear Discriminant" :
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        da = LinearDiscriminantAnalysis()
        model = da.fit(train_df_nolabel.values,train_label)
    elif technique == "Gradient Boosting" :
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier()
        model = gb.fit(train_df_nolabel.values,train_label)
    elif technique == "Multi-layer Preceptron":
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier()
        model = mlp.fit(train_df_nolabel.values,train_label)
    elif technique == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        lr= LogisticRegression()
        model = lr.fit(train_df_nolabel.values,train_label)
    else:
        print("typing errro no such technique:"+ technique)

    return model

def predict_test(test_df,model):
    test_df_copy = test_df.copy()
    test_df_nolabel = test_df.iloc[:, :-1]
    test_label = test_df.iloc[:,-1]
    y_pred = model.predict(test_df_nolabel)
    return test_df_nolabel,test_label,y_pred,model

def score(test_result,testlabel):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(testlabel,test_result)
    print("accuracy :",accuracy)
    from sklearn.metrics import precision_score
    prescision = precision_score(testlabel,test_result)
    print("prescision:",prescision)
    from sklearn.metrics import recall_score
    recall = recall_score(testlabel,test_result)
    print("recall:",recall)
    from sklearn.metrics import f1_score
    f1 = f1_score(testlabel,test_result)
    print("f1:",f1)
    from sklearn.metrics import roc_auc_score
    rauc = roc_auc_score(testlabel,test_result)
    print("rauc:",rauc)





if __name__ == '__main__':
    train_df,test_df = onload()
    train_label = train_df.columns[len(train_df.columns)-1]
    test_label = test_df.columns[len(test_df.columns)-1]
    #****************************
    #**************kNN***********

    technique ="KNN";
    print("********" + technique + "***********")
    model = build_model(train_df,train_label,technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df,model)
    score(y_pred,test_df_label)
    #***************************
    technique="Naive Bayes"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    #***************************
    technique="SVM"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    # ***************************
    technique="Decesion Tree"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    #***************************
    technique="Random Forest"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    #***************************
    technique="AdaBoost"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    #***************************
    technique="Gradient Boosting"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    #***************************
    technique="Linear Discriminant"
    print("*********"+ technique+"***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    # ***************************
    technique = "Multi-layer Preceptron"
    print("*********" + technique + "***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)
    # ***************************
    technique = "Logistic Regression"
    print("*********" + technique + "***********")
    model = build_model(train_df, test_df, technique)
    test_df_nolabel, test_df_label, y_pred, model = predict_test(test_df, model)
    score(y_pred, test_df_label)


    # build_model(train_df,test_df)