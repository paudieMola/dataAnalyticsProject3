#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on a windy day, unlike australia

@author: Helen Daly
@id: R00142752
@Cohort: SOFT8032_26756
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rnd

from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
    
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("weatherAUS.csv", encoding = 'utf8')



def Task1():
    
    # Use 3 attributes to use machine learning to predict if it will rain tomorrow. 
    # Experiment with 1 - 35 depth of decision tree. 
    
    #print(df.info())
    newdf = df[['MinTemp', 'WindGustSpeed', 'Rainfall', 'RainTomorrow']].copy()
    #print(newdf.info())
    newdf = newdf.dropna()
    
    def testTrainPlot(X, y, df):
        
        attributes = list(df.columns)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
        trainResult = []
        testResult = []
        
        for maxdepthNo in range(1,36):
            tree_clf = tree.DecisionTreeClassifier(max_depth=maxdepthNo)
            tree_clf.fit(X_train, y_train)
            trainResult.append(tree_clf.score(X_train, y_train))
            testResult.append(tree_clf.score(X_test, y_test))
        
        title = ''
        for colTitle in attributes:
            title = title + colTitle + ' + '
        title = title + 'used to predict ' + y.columns
        plt.title(title)
        plt.plot(trainResult)
        plt.plot(testResult)
        plt.legend(['Train', 'Test'])
        plt.show()
    
    newdf1 = newdf[['MinTemp', 'WindGustSpeed', 'Rainfall']]
    newdf2 = newdf[['MinTemp', 'WindGustSpeed']]
    newdf3 = newdf[['MinTemp', 'Rainfall']]
    newdf4 = newdf[['WindGustSpeed', 'Rainfall']]
    
    
    X = (newdf1)
    y = newdf[['RainTomorrow']]
    testTrainPlot(X, y, newdf1)
    
    X = (newdf2)
    testTrainPlot(X, y, newdf2)
    
    X = (newdf3)
    testTrainPlot(X, y, newdf3)
    
    X = (newdf4)
    testTrainPlot(X, y, newdf4)


def Task2():
    newdf = df[['Pressure9am', 'Pressure3pm', 'Humidity9am', 'Humidity3pm', 'RainToday']]
    print(newdf.head)
    # To get an idea of how many null values in columns
    print(newdf.info())
    # To fill in blanks for pressure and humidity, to ensure average is correct
    #newdf.loc[newdf['Pressure9am'].isnull(), 'Pressure9am'] = newdf['Pressure3pm']
    #newdf.loc[newdf['Pressure3pm'].isnull(), 'Pressure3pm'] = newdf['Pressure9am']
    #newdf.loc[newdf['Humidity9am'].isnull(), 'Humidity9am'] = newdf['Humidity3pm']
    #newdf.loc[newdf['Humidity3pm'].isnull(), 'Humidity3pm'] = newdf['Humidity9am']
    
    # Fill up the remaining null values with average
    #newdf.loc[newdf['Pressure9am'].isnull(), 'Pressure9am'] = newdf['Pressure9am'].mean()
    #newdf.loc[newdf['Pressure3pm'].isnull(), 'Pressure3pm'] = newdf['Pressure3pm'].mean()
    #print(newdf.info())
    
    #newdf.dropna(subset=['RainToday'], inplace = True)
    #newdf.loc[newdf['Humidity9am'].isnull(), 'Humidity9am'] = newdf['Humidity9am'].mean()
    #newdf.loc[newdf['Humidity3pm'].isnull(), 'Humidity3pm'] = newdf['Humidity3pm'].mean()
    #print(newdf.info())
    
    newdf = newdf.dropna()
    newdf['Pressure'] = (newdf['Pressure9am'] + newdf['Pressure3pm'])/2
    newdf['Humidity'] = (newdf['Humidity9am'] + newdf['Humidity3pm'])/2
    print(newdf.info())
    
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DTC', tree.DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(kernel='linear')))
    models.append(('RFS',RandomForestClassifier()))
    
    #names = []
    results = {}
    
    X = newdf[['Pressure', 'Humidity']]
    y = newdf[['RainToday']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    
    for name,model in models:
        cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy', return_train_score=True)
        results[name] = cv_results
        print(cv_results)
    # for models in results:
    #     print(models)
    #     print('Training  ',results[models]['train_score'].mean())
    #     print('Test  ',results[models]['test_score'].mean())
    # return results
        
    df = pd.DataFrame()

    for models in results:
    
        print(models)
        model = []
        model.append(models)
        data = []
        print('Training  ',results[models]['train_score'].mean())
        data.append(results[models]['train_score'].mean())
        print('Test  ',results[models]['test_score'].mean())
        data.append(results[models]['test_score'].mean())
        ser = pd.Series(data=results[models])
        print(ser)
        df[models] = data

    df = df.transpose()


    df.columns = ['Test Score', 'Train Score']
   
    df.plot.bar(rot=0)       
        
    

def Task3():
    
    return
    
    
    
    
    

def Task4():
    return
   
results = Task2()





    
