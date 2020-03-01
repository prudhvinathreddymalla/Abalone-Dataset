# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 01:16:42 2020

@author: Prudhvinath
"""
#%% import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
plt.savefig('./Results/'+ "Correlation between variables")
#checking for missing values
df1.isna().sum() 
df1.info()

#plotting a pair plot to check
plt.figure(2, figsize = (12, 10))
sns.pairplot(df1)
plt.savefig('./Results/'+ "Pair Plot", dpi = 400)

#we have one categorical variable. ('Sex')
#plotting to check how many categories for 'Sex' variable
sns.countplot(df1.Sex)

#exploring the sex columns more
plt.figure(3, figsize=(12, 10))
g = sns.FacetGrid(data = df1, col= 'Sex', height = 4)
g.map(sns.distplot, 'Rings')
plt.savefig('./Results/'+ "Sex dist Plot", dpi = 400)

# dummy columns are created for the categories in Sex
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['Sex'] = le.fit_transform(df1.Sex)
 #the dummy columns are included in df now

#as per description age = number of rings + 1.5
df1['Age'] = df1.Rings + 1.5
df1['Age'].describe()
#lets check the age plot
plt.figure(4, figsize=(10, 6))
sns.countplot(df1['Age'])
plt.savefig('./Results/'+ "Age Count Plot", dpi = 400)

AgeValues = df1['Age'].values
AgeIndex = []
# 0 is young, 1 is old
for age in AgeValues:
    if age <8:
        AgeIndex.append('0')
    else:
        AgeIndex.append('1')

AgeIndex = pd.DataFrame(data = AgeIndex, columns = ['AgeIndex'])
df1.reset_index(drop=True, inplace=True)
AgeIndex.reset_index(drop = True, inplace = True)
newDf = pd.concat([df1, AgeIndex], axis = 1)

plt.figure(5)
sns.countplot(newDf['AgeIndex'])
plt.savefig('./Results/'+ "AgeIndex Count Plot", dpi = 400)

newDf.drop(['AgeIndex', 'Sex'], axis = 1, inplace = True)
y = AgeIndex.values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(newDf)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size=0.4)


#%% Classification Model

from sklearn.svm import SVC
svcModel = SVC()
svcModel.fit(X_train, y_train)

y_pred = svcModel.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
accuracyScore = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)

#grid search to find the best parameters
from sklearn.model_selection import GridSearchCV
params = {'C': [0.001, 0.01, 0.1, 1, 10],'gamma':[0.001, 0.01, 0.1, 1]}
svcClf = GridSearchCV(svcModel, param_grid = params, scoring = 'accuracy', cv = 10)
svcClf.fit(X_train, y_train)
#
# best parameters and also the best score
print("Tuned Linear Regression Parameters: {}".format(svcClf.best_params_))
print("Best score is {}".format(svcClf.best_score_))

#kfold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svcModel, X, y, cv=10) 
avgScore = scores.mean()
print("The Average Accuracy Score of the model with 10k folds", avgScore)

                     







        


