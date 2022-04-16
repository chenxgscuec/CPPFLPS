# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:57:28 2022

@author: sxg
"""

import numpy as np
from numpy import array
#import pickle
import pandas as pd
import pymrmr
#import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

def model_performance(predicted_probas,y_pred,y_test):  
    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0][0]/float(cm[0][0]+cm[0][1])
    fpr, tpr, auc_thresholds = roc_curve(y_test, predicted_probas )
    auc_final = auc(fpr, tpr)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    [precision, recall, fbeta_score, support ] = precision_recall_fscore_support(y_test, y_pred)
    metrics = [specificity, recall[1], precision[1], accuracy, fbeta_score[1],auc_final, mcc]
    return metrics

class_sel = 'num16'

dataframe = pd.read_csv('train_'+ class_sel +'.csv',encoding="utf-8")

df_test1 = pd.read_csv('test1_'+ class_sel +'.csv',encoding="utf-8")

df_test2 = pd.read_csv('test2_'+ class_sel +'.csv',encoding="utf-8")

df_test3 = pd.read_csv('test3_'+ class_sel +'.csv',encoding="utf-8")

df_test4 = pd.read_csv('test4_'+ class_sel +'.csv',encoding="utf-8")

feats = pymrmr.mRMR(dataframe, 'MIQ', 3)


# 4. train a new model
XFR_train = np.array(dataframe[feats])
y_train = np.array(dataframe['class'])

X1_test = np.array(df_test1[feats])
y1_test = np.array(df_test1['class'])

X2_test = np.array(df_test2[feats])
y2_test = np.array(df_test2['class'])

X3_test = np.array(df_test3[feats])
y3_test = np.array(df_test3['class'])

X4_test = np.array(df_test4[feats])
y4_test = np.array(df_test4['class'])


#clf = svm.SVC(probability = True, random_state=0)
#clf = MLPClassifier(random_state=0)
#clf = DecisionTreeClassifier(random_state=0)
#clf = RandomForestClassifier(random_state=0)
#clf = ExtraTreesClassifier(random_state=0)
clf = XGBClassifier(random_state=0)

classifier = clf.fit(XFR_train, y_train)

# 5. evaluation
y_pred = classifier.predict(X1_test)
predicted_probas = classifier.predict_proba(X1_test)[:, 1]
metrics1 = model_performance(predicted_probas,y_pred,y1_test)
print(metrics1)

y_pred = classifier.predict(X2_test)
predicted_probas = classifier.predict_proba(X2_test)[:, 1]
metrics2 = model_performance(predicted_probas,y_pred,y2_test)
print(metrics2)

y_pred = classifier.predict(X3_test)
predicted_probas = classifier.predict_proba(X3_test)[:, 1]
metrics3 = model_performance(predicted_probas,y_pred,y3_test)
print(metrics3)

y_pred = classifier.predict(X4_test)
predicted_probas = classifier.predict_proba(X4_test)[:, 1]
metrics4 = model_performance(predicted_probas,y_pred,y4_test)
print(metrics4)

#f = open('result.csv', 'a', encoding='utf-8', newline='')
#wr = csv.writer(f)
#wr.writerow([metrics1])
#wr.writerow([metrics2])
#wr.writerow([metrics3])
#wr.writerow([metrics4])
#f.close()