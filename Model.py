""" Importing libraries"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
import pickle

""" Loading Datasets"""

df = pd.read_csv('dataset1.csv')

""" Eliminating NaN values """

df=df.iloc[0:8384,0:63]

""" Naming columns """

df.columns = [i for i in range(df.shape[1])]


""" Renaming Output Column"""

df = df.rename(columns={62: 'Output'})

""" Splitting Features(Inputs) and Labels(Outputs)"""

X = df.iloc[:, :-1]
#print("Features shape =", X.shape)

""" Splitting Features(Inputs) and Labels(Outputs)"""

Y = df.iloc[:, -1]
#print("Labels shape =", Y.shape)


""" Converting Continuous Values into Categorical Values"""

lab = preprocessing.LabelEncoder()
Y = lab.fit_transform(Y)

""" Splitting data in 80:20 ratio"""
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

""" Initialising model """
svm = SVC(C=10, gamma=0.1, kernel='rbf')

""" Training model """
svm.fit(x_train, y_train)


""" Saving model """
with open('model1.pkl','wb') as f:
    pickle.dump(svm,f)
