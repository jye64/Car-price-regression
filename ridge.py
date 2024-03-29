
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

# ===================== Part 1: Read Dataset =====================
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# ===================== Part 2: EDA and Preprocessing =====================
# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']

# standard Scaler to scale X
scalerX = StandardScaler().fit(X)
X_std = scalerX.transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

# split training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.2, random_state=0)


# ===================== Part 3: Modeling =====================
# Ridge Regression
regr = Ridge()
regr.fit(X_train, Y_train)

results = X_test.copy()
results["predicted"] = regr.predict(X_test)
results["actual"] = Y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)


# ===================== Part 4: Accuracy & Evaluation =====================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('Train Set R^2 Score: ' + str(regr.score(X_train, Y_train)))
print('Train Set MAE: ' + str(mean_absolute_error(Y_train, regr.predict(X_train)).round(2)))
print('Train Set MAPE: ' + str(mean_absolute_percentage_error(Y_train, regr.predict(X_train)).round(2)) + '%' + '\n')

print('Test Set R^2 Score: ' + str(regr.score(X_test, Y_test)))
print('Test Set MAE: ' + str(mean_absolute_error(Y_test, regr.predict(X_test)).round(2)))
print('Test Set MAPE: ' + str(mean_absolute_percentage_error(Y_test, regr.predict(X_test)).round(2)) + '%' + '\n')

# 5 - fold Cross Validation
# R^2 score
CVScores = cross_val_score(regr, X_std, Y, cv=5)
print('Cross Validation R^2 Score: ' + str(CVScores))

# Mean Absolute Error
CV = cross_validate(regr, X_std, Y, cv=5, scoring='neg_mean_absolute_error')
CV['test_score'] = -CV['test_score']
print('Cross Validation MAE: ' + str(CV['test_score'].round(2)))
print('Cross Validation Overall MAE: ' + str(np.mean(CV['test_score']).round(2)))

