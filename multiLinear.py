
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ===================== Part 1: Read Dataset =====================
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# ===================== Part 2: EDA and Preprocessing =====================
# print(data.head())
# print(data.isnull().sum())
# print(data.describe())

# sns.countplot(x="transmission", data=data)
# plt.show()
#
# print(data["model"].value_counts() / len(data))
# sns.countplot(y=data["model"])
# plt.show()
#
# sns.countplot(x="fuelType", data=data)
# plt.show()

# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

# standard Scaler to scale all variables
std = StandardScaler()
data_onehot_std = std.fit_transform(data_onehot)
data_onehot_std = pd.DataFrame(data_onehot_std, columns=data_onehot.columns)

X = data_onehot_std.drop(['price'], axis=1)
Y = data_onehot_std[['price']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)

# TODO: K-fold cross validation
# kf = KFold(n_splits=5)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
print(regr.score(X, Y))

X_test = std.inverse_transform(X_test)

# prediction
results = X_test.copy()
results["predicted"] = regr.predict(X_test)
results["actual"] = Y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)
