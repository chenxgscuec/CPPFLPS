# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 08:30:42 2021

@author: sxg
"""
import numpy as np
from numpy import array
import pickle

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

def test(save_path,classifier,mean,std): 
#	save_path2 = 'datatest1.pkl'
	datate = pickle.load(open(save_path, 'rb'))
	X_pos = datate['X_pos']
	X_neg = datate['X_neg']   
	X_pos = np.array(X_pos)
	X_neg = np.array(X_neg)
	X_test = np.concatenate((X_pos,X_neg),axis = 0) 
	y_test = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
	X_test = X_test - mean
	X_test = X_test/std
    
	y_pred = classifier.predict(X_test)
	predicted_probas = classifier.predict_proba(X_test)[:, 1]
	metrics = model_performance(predicted_probas,y_pred,y_test)
	print(metrics)
    
save_path1 = 'datatrain.pkl'
datatr = pickle.load(open(save_path1, 'rb'))

X_pos = datatr['X_pos']

X_neg = datatr['X_neg']

X_pos = np.array(X_pos)
X_neg = np.array(X_neg)

y_train = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])

X_train = np.concatenate((X_pos,X_neg),axis = 0)
scaler = StandardScaler()
X_train = scaler.fit(X_train).transform(X_train)
mean = scaler.mean_
std = np.sqrt(scaler.var_)

#clf = svm.SVC(probability = True, random_state=0)
#clf = RandomForestClassifier(random_state=0)
#clf = ExtraTreesClassifier(random_state=0)
clf = XGBClassifier(random_state=0)

classifier = clf.fit(X_train, y_train)
save_path1 = 'datatest1.pkl'
test(save_path1,classifier,mean,std)
save_path2 = 'datatest2.pkl'
test(save_path2,classifier,mean,std)
save_path3 = 'datatest3.pkl'
test(save_path3,classifier,mean,std)
save_path4 = 'datatest4.pkl'
test(save_path4,classifier,mean,std)