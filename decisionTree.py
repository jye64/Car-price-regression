
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# ===================== Part 1: Read Dataset =====================
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# ===================== Part 2: EDA and Preprocessing =====================
print(data.isnull().sum())
print(data.describe())

sns.countplot(x="transmission", data=data)
plt.show()

sns.countplot(y=data["model"])
plt.show()

sns.countplot(x="fuelType", data=data)
plt.show()

sns.pairplot(data)
plt.show()

# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

# separate features and target variable
X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']

# split training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)

# standard Scaler to fit training data in X
scalerX = StandardScaler().fit(X_train)
X_train_std = scalerX.transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

# apply the same scaler on testing data
X_test_std = scalerX.transform(X_test)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)


# ===================== Part 3: Modeling =====================
# Decision Tree Regression
regr = DecisionTreeRegressor()
regr.fit(X_train_std, Y_train)

results = X_test.copy()
results["predicted"] = regr.predict(X_test_std)
results["actual"] = Y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)

del results


# ===================== Part 4: Accuracy & Evaluation =====================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('Train Set MAE: ' + str(mean_absolute_error(Y_train, regr.predict(X_train_std)).round(2)))
print('Train Set RMSE: ' + str(np.sqrt(mean_squared_error(Y_train, regr.predict(X_train_std))).round(2)))
print('Train Set MAPE: ' + str(mean_absolute_percentage_error(Y_train, regr.predict(X_train_std)).round(2)) + '%' + '\n')

print('Test Set MAE: ' + str(mean_absolute_error(Y_test, regr.predict(X_test_std)).round(2)))
print('Test Set RMSE: ' + str(np.sqrt(mean_squared_error(Y_test, regr.predict(X_test_std))).round(2)))
print('Test Set MAPE: ' + str(mean_absolute_percentage_error(Y_test, regr.predict(X_test_std)).round(2)) + '%' + '\n')


# 5 - fold Cross Validation
# RMSE
pipe = make_pipeline(StandardScaler(), regr)
CV = cross_validate(pipe, X, Y, cv=5, scoring='neg_root_mean_squared_error')
CV['test_score'] = -CV['test_score']
print('Cross Validation RMSE: ' + str(CV['test_score'].round(2)))
print('Cross Validation Overall RMSE: ' + str(np.mean(CV['test_score']).round(2)))

