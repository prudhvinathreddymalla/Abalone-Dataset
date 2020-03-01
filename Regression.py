# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:06:50 2020

@author: Prudhvinath
"""

#%% import the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

seed = np.random.randint(100)
#83
np.random.seed(seed)

import warnings
warnings.filterwarnings('ignore')
#%% import the dataset (Ablone dataset)

df = pd.read_csv('abalone.csv')

#%% Data preprocessing

df.describe()
df['Height'].describe()
#There are two zeros in height. Since it does not make any sense, removing those complete rows
df[df.Height == 0]
df1 = df[df.Height != 0]
df1.describe()
df1['Height'].describe()
df1.corr()
#plotting the correlation
plt.figure(1)
sns.heatmap(df.corr(), annot = True)
# plt.savefig('./Results/'+ "Correlation between variables for regression", dpi = 400)
#checking for missing values
df1.isna().sum() 
df1.info()

#plotting a pair plot to check
plt.figure(2, figsize = (12, 10))
sns.pairplot(df1)
# plt.savefig('./Results/'+ "Pair Plot for regression", dpi = 400)

#we have one categorical variable. ('Sex')
#plotting to check how many categories for 'Sex' variable
sns.countplot(df1.Sex)

#exploring the sex columns more
plt.figure(3, figsize=(12, 10))
g = sns.FacetGrid(data = df1, col= 'Sex', height = 4)
g.map(sns.distplot, 'Rings')
# plt.savefig('./Results/'+ "Sex dist Plot for regression", dpi = 400)

# dummy columns are created for the categories in Sex
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['Sex'] = le.fit_transform(df1.Sex)

#as per descripton age = rings + 1.5
df1['Age'] = df.Rings + 1.5
df1.groupby('Sex')['Rings']


X = df.drop(['Rings', 'Sex'], axis = 1)
y = df[['Rings']]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = seed)

#%% Regression model (linear Regressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

regModel = LinearRegression()
regModel.fit(X_train, y_train)

y_pred_train = regModel.predict(X_train)
#evaluating model on train set
kfold = KFold(n_splits = 5, random_state = seed)
cv_results = cross_val_score(regModel, X_train, y_train, scoring='neg_mean_squared_error', cv = kfold)


#this fucntion is made if to use multiple model. but here we are using only linear regression model
def predictionPlot(model, plotTraining = False):
    """
    fit model to test set
    makes predictions on test set
    prints metrics
    plots the predictions vs True values
    """
    #making predictions on test set
    y_pred_train = regModel.predict(X_train)
    y_pred = regModel.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error', mae)
    mse= mean_squared_error(y_test, y_pred)
    print('Mean Squared Error', mse)
    r2Score = r2_score(y_test, y_pred)
    print('R2 Score', r2Score)
    if plotTraining:
        plt.figure(4)
        plt.scatter(y_train, y_pred_train, label = 'Training Data')
    plt.figure(5)
    plt.scatter(y_test, y_pred, label = 'Testing Data')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title('{}\n Predicted vs Actual'.format(str(type(regModel)).replace('.', ' ').replace('>', ' ').replace("'", ' ').split(' ')[-3]))
    plt.legend()
    plt.draw()
    # plt.savefig('./Results/'+ " Regression Plot", dpi = 400)


predictionPlot(regModel, plotTraining=True)        
