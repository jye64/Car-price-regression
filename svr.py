
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# read dataset
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])
X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']

# split train and test sets
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)

# SVR
regr = make_pipeline(StandardScaler(),SVR(kernel='linear', C=1.0, epsilon=0.2))
regr.fit(X_train,Y_train)
print(regr.score(X,Y))

# prediction
results = X_test.copy()
results["predicted"] = regr.predict(X_test)
results["actual"]= Y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)

