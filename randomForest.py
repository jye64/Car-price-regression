
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# ===================== Part 1: Read Dataset =====================
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']

# standard Scaler to scale X
scalerX = StandardScaler().fit(X)
X_std = scalerX.transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

# 5 - fold Cross Validation
regr = RandomForestRegressor(random_state=1)
CVScores = cross_val_score(regr, X_std, Y, cv=5)
print('Cross Validation Score: ' + str(CVScores))

# prediction
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.2, random_state=25)
regr.fit(X_train, Y_train)
print('Test Set Score: ' + str(regr.score(X_test, Y_test)))

results = X_test.copy()
results["predicted"] = regr.predict(X_test)
results["actual"] = Y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
print(results)


