
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
from sklearn.model_selection import cross_val_score

# ===================== Part 1: Read Dataset =====================
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# ===================== Part 2: EDA =====================
# print(data.head())
# print(data.isnull().sum())
# print(data.describe())

# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']

# SVR
regr = make_pipeline(StandardScaler(),SVR(kernel='linear', C=1.0, epsilon=0.2))
CVScores = cross_val_score(regr, X, Y, cv=5)
print('Cross Validation Score: ' + str(CVScores))

# prediction
# split train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
regr.fit(X_train,Y_train)
print(regr.score(X_test,Y_test))

results = X_test.copy()
results["predicted"] = regr.predict(X_test)
results["actual"]= Y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)

