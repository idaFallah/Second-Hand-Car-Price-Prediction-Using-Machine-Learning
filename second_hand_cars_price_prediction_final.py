
# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset
dataset = pd.read_csv('/content/car data.csv')

dataset.head()

# exploring the data
dataset.shape

dataset.columns

dataset.info()

# categrical data
dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

# numerical data
dataset.select_dtypes(include=['int64', 'float64']).columns

len(dataset.select_dtypes(include=['int64', 'float64']).columns)

# statistical summary
dataset.describe()

#dealing with null values
dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[dataset.isnull().any()]

len(dataset.columns[dataset.isnull().any()])  # no null values

# reconstructing the dataset
dataset = dataset.drop('Car_Name', axis=1)

dataset.head()

dataset['Current_Year'] = 2023 # add a column

dataset.head()

dataset['Years_Used'] = dataset['Current_Year'] - dataset['Year']

dataset.head()

dataset = dataset.drop(['Year', 'Current_Year'], axis=1)

dataset.head()

# encoding categorical data

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

dataset['Fuel_Type'].nunique()

dataset['Seller_Type'].nunique()

dataset['Transmission'].nunique()

dataset.shape

# one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)

dataset.head()

dataset.shape

# correlation matrix

dataset_2 = dataset.drop('Selling_Price', axis=1)

dataset_2.corrwith(dataset['Selling_Price']).plot.bar(
    figsize=(16, 9), title='Correlation with Selling Price',
    rot=45, grid=True
)

corr = dataset.corr()

# heatmap
plt.figure(figsize=(16, 9))
sns.heatmap(corr,cmap='coolwarm', annot=True)

# splitting the dataset

dataset.head()

# matrix of features/independant variables
x = dataset.drop('Selling_Price', axis=1)

#dependant variable/ target value
y = dataset['Selling_Price']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

y_train.shape

x_test.shape

y_test.shape

# feature scaling

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

# we're not using it on this dataset, since we're only gonna use 2 models: Multi LR & RF

# building the model

from sklearn.linear_model import LinearRegression
regressor_MLR = LinearRegression()
regressor_MLR.fit(x_train, y_train)

y_pred = regressor_MLR.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

# randm forest
from sklearn.ensemble import RandomForestRegressor
regressor_RF = RandomForestRegressor()
regressor_RF.fit(x_train, y_train)

y_pred_RF = regressor_RF.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred_RF)

# finding the optimal params by use of randomized search
from sklearn.model_selection import RandomizedSearchCV

parameters = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [10, 20, 30, 40, 50],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20]
}

parameters

random_cv = RandomizedSearchCV(estimator=regressor_RF, param_distributions=parameters, n_iter=10, scoring= 'neg_mean_absolute_error', cv=5, verbose=2, n_jobs=-1, error_score='raise')

random_cv.fit(x_train, y_train)

random_cv.best_estimator_

random_cv.best_params_

# final model : random forest

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features='log2', max_depth=10, criterion='absolute_error')
regressor.fit(x_train, y_train)

y_pred_final = regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred_final)

# predicting a single observation

dataset.head()

single_obs = [8.5, 45000, 0, 1, 0, 0, 1, 0]

regressor.predict([single_obs])





