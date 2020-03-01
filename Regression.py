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
