
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


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

# correlation matrix
corr_matrix = data.corr()
print(corr_matrix['price'].sort_values(ascending=False))

# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

# separate features and target variable
X = data_onehot.drop(['price'], axis=1)
Y = data_onehot['price']

# split training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# standard Scaler to fit training data in X
scalerX = StandardScaler().fit(X_train)
X_train_std = scalerX.transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

# apply the same scaler on testing data
X_test_std = scalerX.transform(X_test)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)


# ===================== Part 3: Modeling =====================

models_to_evaluate = [DecisionTreeRegressor(), SVR(kernel='linear'), RandomForestRegressor()]

for model in models_to_evaluate:
    regr = model
    model_name = str(model)
    regr.fit(X_train_std, Y_train)

    # copy the DataFrame indexes
    results = X_train.copy()
    results["predicted"] = regr.predict(X_train_std)
    results["actual"] = Y_train
    results = results[['predicted', 'actual']]
    results['predicted'] = results['predicted'].round(2)

    # reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
    results = pd.DataFrame.reset_index(results, drop=True)

    # visualize predicted vs actual in train set
    plt.plot(results['predicted'].head(100), label='predicted')
    plt.plot(results['actual'].head(100), label='actual')
    plt.xlabel('index in train set')
    plt.ylabel('price')
    plt.title(model_name + ':Predicted vs Actual in Train set')
    plt.legend()
    plt.show()


# ===================== Part 4: Accuracy & Evaluation =====================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


model_performance = pd.DataFrame(columns=['Model', 'Train MAE', 'Train RMSE', 'Train MAPE',
                                          'CV RMSE', 'Test MAE', 'Test RMSE', 'Test MAPE'])

for model in models_to_evaluate:
    regr = model
    model_name = str(model)

    Train_MAE = mean_absolute_error(Y_train, regr.predict(X_train_std)).round(2)
    Train_RMSE = np.sqrt(mean_squared_error(Y_train, regr.predict(X_train_std))).round(2)
    Train_MAPE = mean_absolute_percentage_error(Y_train, regr.predict(X_train_std)).round(2)

    # 5 - fold Cross Validation on training data for model validation
    # RMSE
    CV = cross_validate(regr, X_train_std, Y_train, cv=5, scoring='neg_root_mean_squared_error')
    CV['test_score'] = -CV['test_score']
    CV_Overall_RMSE = np.mean(CV['test_score']).round(2)

    # after validating the model, use the test set to compute generalization error
    Test_MAE = mean_absolute_error(Y_test, regr.predict(X_test_std)).round(2)
    Test_RMSE = np.sqrt(mean_squared_error(Y_test, regr.predict(X_test_std))).round(2)
    Test_MAPE = mean_absolute_percentage_error(Y_test, regr.predict(X_test_std)).round(2)

    model_performance = model_performance.append({'Model': model_name, 'Train MAE': Train_MAE,
                                                  'Train RMSE': Train_RMSE, 'Train MAPE': Train_MAPE,
                                                  'CV RMSE': CV_Overall_RMSE, 'Test MAE': Test_MAE,
                                                  'Test RMSE': Test_RMSE, 'Test MAPE': Test_MAPE},
                                                 ignore_index=True)

    # copy the DataFrame indexes
    test_results = X_test.copy()
    test_results["predicted"] = regr.predict(X_test_std)
    test_results["actual"] = Y_test
    test_results = test_results[['predicted', 'actual']]
    test_results['predicted'] = test_results['predicted'].round(2)

    # reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
    test_results = pd.DataFrame.reset_index(test_results, drop=True)

    # visualize predicted vs actual in test set
    plt.plot(test_results['predicted'].head(100), label='predicted')
    plt.plot(test_results['actual'].head(100), label='actual')
    plt.xlabel('index in test set')
    plt.ylabel('price')
    plt.title(model_name + ':Predicted vs Actual in test set')
    plt.legend()
    plt.show()

pd.set_option('max_columns', 8)
print(model_performance)
