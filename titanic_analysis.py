# Python 2.7.9

import numpy as np
import pandas

train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

train["Fare"] = train["Fare"].fillna(train["Fare"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
train["Embarked"] = train["Embarked"].fillna("S") # S is most common location
test["Embarked"] = test["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

train_features = train[['Age','Sex','Pclass','Parch','SibSp','Fare','Embarked']].astype(float)
train_labels = train["Survived"]
test_features = test[["Age",'Sex','Pclass','Parch','SibSp','Fare','Embarked']].astype(float)

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, min_samples_split=20, min_samples_leaf=20)  #with ~800 rows, this is about 2% of the rows
clf = clf.fit(train_features, train_labels)
train_predict = clf.predict(train_features)


from sklearn.metrics import *
score = accuracy_score(train_labels, train_predict)
precision = precision_score(train_labels, train_predict)
recall = recall_score(train_labels, train_predict)
print " for our training set "
print score, " is our accuracy "
print precision, " is our precision"
print recall, " is our recall"


test_predict = clf.predict(test_features)
df = pandas.DataFrame(test['PassengerId'])
df2 = pandas.DataFrame(test_predict)
df = df.join(df2)
df.columns = ['PassengerId','Survived']

df.to_csv("test_predict.csv",index=False)
