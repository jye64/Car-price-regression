
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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
print(data)

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

# separate features and target variable
X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']


# split train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# standard Scaler to fit training data in X
scalerX = StandardScaler().fit(X_train)
X_train_std = scalerX.transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

# apply the same scaler on testing data
X_test_std = scalerX.transform(X_test)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)


# ===================== Part 3: Modeling =====================
# Support vector regression
regr = SVR(kernel='linear', C=1.0, epsilon=0.1)
regr.fit(X_train_std, Y_train)

# copy the DataFrame indexes
results = X_train.copy()
results["predicted"] = regr.predict(X_train_std)
results["actual"] = Y_train
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)

# reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
results = pd.DataFrame.reset_index(results, drop=True)

# visualize predicted vs actual in train set
plt.plot(results['predicted'].head(100), label='predicted')
plt.plot(results['actual'].head(100), label='actual')
plt.xlabel('index in train set')
plt.ylabel('price')
plt.title('SVR: Predicted vs Actual in Train set')
plt.legend()
plt.show()


# ===================== Part 4: Accuracy & Evaluation =====================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('Train Set MAE: ' + str(mean_absolute_error(Y_train, regr.predict(X_train_std)).round(2)))
print('Train Set RMSE: ' + str(np.sqrt(mean_squared_error(Y_train, regr.predict(X_train_std))).round(2)))
print('Train Set MAPE: ' + str(mean_absolute_percentage_error(Y_train, regr.predict(X_train_std)).round(2)) + '%' + '\n')


# 5 - fold Cross Validation
# RMSE
CV = cross_validate(regr, X_train_std, Y_train, cv=5, scoring='neg_root_mean_squared_error')
CV['test_score'] = -CV['test_score']
print('Cross Validation RMSE: ' + str(CV['test_score'].round(2)))
print('Cross Validation Overall RMSE: ' + str(np.mean(CV['test_score']).round(2)) + '\n')


# after validating the model, use the test set to compute generalization error
print('Test Set MAE: ' + str(mean_absolute_error(Y_test, regr.predict(X_test_std)).round(2)))
print('Test Set RMSE: ' + str(np.sqrt(mean_squared_error(Y_test, regr.predict(X_test_std))).round(2)))
print('Test Set MAPE: ' + str(mean_absolute_percentage_error(Y_test, regr.predict(X_test_std)).round(2)) + '%' + '\n')


# copy the DataFrame indexes
test_results = X_test.copy()
test_results["predicted"] = regr.predict(X_test_std)
test_results["actual"] = Y_test
test_results = test_results[['predicted', 'actual']]
test_results['predicted'] = test_results['predicted'].round(2)
print(test_results)

# reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
test_results = pd.DataFrame.reset_index(test_results, drop=True)

# visualize predicted vs actual in test set
plt.plot(test_results['predicted'].head(100), label='predicted')
plt.plot(test_results['actual'].head(100), label='actual')
plt.xlabel('index in test set')
plt.ylabel('price')
plt.title('SVR: Predicted vs Actual in test set')
plt.legend()
plt.show()
