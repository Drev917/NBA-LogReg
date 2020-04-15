# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:48:43 2020

@author: Drewb
"""

import pandas as pd
import numpy as np

nbaData = pd.read_csv('NBA201516.csv')

#Get rid of columns we are not using in our model.
del nbaData["Rk"]
del nbaData["Player"]
del nbaData["Age"]
del nbaData["Tm"]
del nbaData["G"]
del nbaData["GS"]
del nbaData["FG"]
del nbaData["FGA"]
del nbaData["FG%"]
del nbaData["3P"]
del nbaData["3PA"]
del nbaData["3P%"]
del nbaData["2P"]
del nbaData["2PA"]
del nbaData["2P%"]
del nbaData["eFG%"]
del nbaData["FT"]
del nbaData["FTA"]
del nbaData["FT%"]
del nbaData["ORB"]
del nbaData["DRB"]
del nbaData["BLK"]
del nbaData["TOV"]
del nbaData["PF"]

nbaData["Good Rebounder"] = np.where(nbaData["TRB"]>8, 'Yes', 'No') #Defining Good Rebounder Column
nbaData = nbaData[["Good Rebounder","MP","Pos","PS/G","AST","STL"]] #5 predictor columns (x1,x2,x3,x4,x5) and 1 response column (y)
nbaData["Pos"] = nbaData["Pos"].str[:2] #creates the 5 distict Pos values. Deprecates '-'

#transforming categorical data into numerical using indicator variables
nbaData["RebounderNumeric"] = (nbaData["Good Rebounder"]=='Yes')
nbaData["RebounderNumeric"] = (nbaData["Good Rebounder"]=='Yes').astype(int)
del nbaData["Good Rebounder"]
nbaData["Pos"] = nbaData["Pos"].astype('category')
nbaData["Pos"] = nbaData["Pos"].cat.codes
#PG = 2, SG = 4, SF = 3, C = 0, PF = 1

nbaData = nbaData.dropna(axis=0, how='any') #drop any rows containing an NaN
nbaData = nbaData[["RebounderNumeric","MP","Pos","PS/G","AST","STL"]]

numberRows = len(nbaData)
randomlyShuffledRows = np.random.permutation(numberRows)
trainingRows = randomlyShuffledRows[0:320] #use first 320 random rows for training
testRows = randomlyShuffledRows[320:] #remaining rows are test set

xTrain = nbaData.iloc[trainingRows,1:6]
yTrain = nbaData.iloc[trainingRows,0]
xTest = nbaData.iloc[testRows,1:6]
yTest = nbaData.iloc[testRows,0]

from sklearn import linear_model
reg = linear_model.LogisticRegression(solver='lbfgs') #Silencing FutureWarning on scikit
reg.fit(xTrain,yTrain)

model_prediction = reg.predict(xTest)
#model_prediction is 1 or 0.

diff = (model_prediction - yTest)

print(sum(abs(diff))) #wrong predictions

print(reg.coef_) #beta values
print(reg.intercept_) #y response

#scores the accuracy of logistic regression prediction.
score = reg.score(xTest,yTest)
print(score)

del nbaData["RebounderNumeric"]
nbaData = nbaData[["Pos","MP","PS/G","AST","STL"]]

person1 = [4, 15, 10, 4, 2] #SG, 15MP, 10PS/G, 4AST, 2STL
person2 = [0, 30, 5, 5, 4] #C, 30MP, 5PS/G, 5AST, 4STL
person3 = [3, 24, 20, 3, 1.8] #SF, 35MP, 20PS/G, 3AST, 1.8STL

newTestSet = np.vstack((person1,person2,person3))
#vertically stacking person1, person2, person3 in 2-D array

goodRebounder = reg.predict_proba(newTestSet) #input type for any sklearn .predict() functions
print(goodRebounder)

print("Probabilities of player being a good rebounder (in order):")
for i in range(3):
    print(goodRebounder[i][1])
