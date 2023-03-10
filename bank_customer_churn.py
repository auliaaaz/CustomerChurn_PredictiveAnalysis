# -*- coding: utf-8 -*-
"""Dicoding Subm:Bank Customer Churn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sFOGS-tEXSFPeUMK41zDJCQVTHqXtDPw

#### Import libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
import os

# load dataset
dataset = pd.read_csv("/content/Bank Customer Churn Prediction.csv")
dataset

"""Get informations from dataset (ex: data type)"""

dataset.info()

"""There are 2 columns with object type (we need to transform it) and the other are integer or float.

drop customer_id column because it is unnecessary (in case we want to classified churn)
"""

dataset = dataset.drop("customer_id", axis=1)
dataset

"""Statistic Decription

Checking unique value for each column to determine it as categorical or not
"""

for uniq in dataset:
  count = dataset[uniq].drop_duplicates().shape[0]
  print("", uniq, ": ", count)

# age, products_number, active_number, credit_card, tenure, and churn are categorical value
dataset[['credit_score', 'age', 'balance', 'estimated_salary']].describe()

"""there is 0 minimum value in balance it look un normal and also for estimated salary minimum only 11.58 dollar

Handle Missing Values

Checking the total number null value of 'balance' column
"""

balance = (dataset.balance==0).sum()
balance

dataset.loc[(dataset['balance']==0)]

"""there is no spesific difference why the balance is 0, then we will fill it  with mean value"""

dataset['balance'] = dataset['balance'].mask(dataset['balance']==0).fillna(dataset['balance'].mean())

balance = (dataset.balance==0).sum()
balance

"""Handle Outliers

Detecting the outliers with box plot
"""

df = dataset[['credit_score', 'age', 'balance', 'estimated_salary']]

for column in df:
  plt.figure()
  sns.boxplot(data=df, x=column)

"""dropping outliers for credit_score column

"""

Q1 = dataset['credit_score'].quantile(0.25)
Q3 = dataset['credit_score'].quantile(0.75)
IQR = Q3-Q1
lower_lim = Q1-1.5*IQR
upper_lim = Q3+1.5*IQR
print("lower limit: ", lower_lim, "and upper limit: ", upper_lim)

outliers_low1 = (dataset['credit_score'] < lower_lim)
outliers_up1 = (dataset['credit_score'] > upper_lim)
dataset = dataset[~(outliers_low1 | outliers_up1)]
dataset.shape

"""dropping outliers for 'balance' column"""

Q1 = dataset['balance'].quantile(0.25)
Q3 = dataset['balance'].quantile(0.75)
IQR = Q3-Q1
lower_lim = Q1-1.5*IQR
upper_lim = Q3+1.5*IQR
print("lower limit: ", lower_lim, "and upper limit: ", upper_lim)

outliers_low2 = (dataset['balance'] < lower_lim)
outliers_up2 = (dataset['balance'] > upper_lim)
dataset = dataset[~(outliers_low2 | outliers_up2)]
dataset.shape

"""Univariate Analysis

Categorical Analysis
"""

categorical_features = ['country', 'gender', 'tenure', 'products_number', 'credit_card', 'active_member', 'churn']
numerical_features = ['credit_score', 'age', 'balance', 'estimated_salary']

"""Checking total and percentage of every column"""

for features in categorical_features:
  count = dataset[features].value_counts()
  percent = 100*dataset[features].value_counts(normalize=True)
  df = pd.DataFrame({'total sample': count, 'percentage': percent.round(1)})
  print(df)

fig, ax = plt.subplots(7, 1, figsize=(5, 30))
for variable, subplot in zip(categorical_features, ax.flatten()):
    sns.countplot(dataset[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)

"""Numerical Features"""

dataset.hist(['credit_score', 'age', 'balance', 'estimated_salary'], figsize=(20, 15))
plt.show()

"""Multivariate Analysis

Categrical Features
"""

categorical_features2 = ['country', 'gender', 'tenure', 'products_number', 'credit_card', 'active_member']

for col in categorical_features2:
  CrosstabResult=pd.crosstab(index=dataset['churn'], columns=dataset[col])
  CrosstabResult.plot.bar()

"""Numerical Features

drop all the categorical features to only plot the numerical features
"""

dataset_num = dataset.drop(categorical_features, axis=1)

sns.pairplot(dataset_num, diag_kind="kde")

"""Data Preprocessing

Encoding categorical features
"""

from sklearn.preprocessing import OneHotEncoder
dataset = pd.concat([dataset, pd.get_dummies(dataset['country'], prefix='country')], axis=1)
dataset = pd.concat([dataset, pd.get_dummies(dataset['gender'], prefix='gender')], axis=1)
dataset.drop(['country', 'gender'], axis=1, inplace=True)
dataset.head()

"""Splitting dataset into train and test data"""

from sklearn.model_selection import train_test_split

X = dataset.drop(["churn"], axis=1)
Y = dataset["churn"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

"""standardize numerical features"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

model1 = RandomForestClassifier(n_estimators = 100, max_depth=5)
# training model
model1.fit(X_train, Y_train)
# performing prediction
pred_model1 = model1.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(Y_test, pred_model1))

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model1, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(Y_test, pred_model1))

"""K-Nearest Neighbor Classifier"""

from sklearn.neighbors import KNeighborsClassifier

# trying model for k=1 through 50 and record testing accuracy
range_k = range(1, 51)
scores = {}
list_score = []
for k in range_k:
  model2 = KNeighborsClassifier(n_neighbors = k)
  model2.fit(X_train, Y_train)
  pred_model2 = model2.predict(X_test)
  scores[k] = metrics.accuracy_score(Y_test, pred_model2)
  list_score.append(metrics.accuracy_score(Y_test, y_pred))

# plot k scores with accuracy
plt.plot(range_k, list_score)
plt.xlabel('K Value')
plt.ylabel('Testing Accuracy')

# we will use for k = 10
model2 = KNeighborsClassifier(n_neighbors=10)
#training model
model2.fit(X_train, Y_train)
# performing prediction
pred_model2 = model2.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(Y_test, pred_model2))

plot_confusion_matrix(model2, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()

print(classification_report(Y_test, pred_model2))

"""Compare and Choosing the Best Model

As we can see, random forest classifier have higher number for accuracy and precision, but the recall score is lower than KNN model. Because of that score and the stabilization (which random forest more stabil than KNN for both 0 and 1 label) then we choose random forest model.

Test dataset
"""

prediction = X_test.iloc[:1].copy()
pred = Y_test[:1]
rforest = model1.predict(prediction)
knn = model2.predict(prediction)
print("", prediction, " y_true:", pred, " random_forest:", rforest, " KNN:", knn)