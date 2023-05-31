# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 08:30:42 2021

@author: sxg
"""
import numpy as np
from numpy import array
import pickle
import pandas as pd
#import pymrmr

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

def array_test(datate,scaler,classifier):
#	save_path2 = 'datatest1.pkl'
#    datate = pickle.load(open(save_path, 'rb'))
    X_pos = datate['X_pos']
    X_neg = datate['X_neg']   
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    X_test = np.concatenate((X_pos,X_neg),axis = 0) 
    #	y_test = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    X_test = X_test - mean
    X_test = X_test/std
    
    y_fea = classifier.predict(X_test)
    y_fea = list(y_fea)
    return y_fea

# 1. construct a feature pool, save as .csv file
data_list = ['da1p1.pkl','da1p2.pkl','da1p3.pkl','da1p4.pkl','da1p5.pkl','da1p6.pkl',
             'da2p1.pkl','da2p2.pkl','da2p3.pkl','da2p4.pkl','da2p5.pkl','da2p6.pkl',
             'da3p1.pkl','da3p2.pkl','da3p3.pkl','da3p4.pkl','da3p5.pkl','da3p6.pkl',
             'da4p1.pkl','da4p2.pkl','da4p3.pkl','da4p4.pkl','da4p5.pkl','da4p6.pkl',
             'da5p1.pkl','da5p2.pkl','da5p3.pkl','da5p4.pkl','da5p5.pkl','da5p6.pkl',
             'da6p1.pkl','da6p2.pkl','da6p3.pkl','da6p4.pkl','da6p5.pkl','da6p6.pkl']

# num = 4
for num in range(2,36,2):
    save_path1 = 'datatrain.pkl'
    datatr = pickle.load(open(save_path1, 'rb'))

    X_pos = datatr['X_pos']
    X_neg = datatr['X_neg']
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)

    X_train0 = np.concatenate((X_pos, X_neg), axis=0)
    y = np.array([1] * X_pos.shape[0] + [0] * X_neg.shape[0])
    y_list = list(y)
    data_dict = {'class': y_list}

    scaler_arr = []
    classifier_arr = []
    for idx in range(len(data_list)):
        data_name = data_list[idx]
        da = pickle.load(open(data_name, 'rb'))
        col_name = data_name[:5]
        X_add = da['X_pos']
    #    p_num = 4
        #print(p_num)
        X_add = X_add[:449*num]
        X_add = np.array(X_add)

        X_pos = np.concatenate((X_pos,X_add),axis = 0)

        y_train = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])

        X_train = np.concatenate((X_pos,X_neg),axis = 0)
        scaler = StandardScaler()
        X_train = scaler.fit(X_train).transform(X_train)
        scaler_arr.append(scaler)
        #clf = svm.SVC(probability = True, random_state=0)
        #clf = MLPClassifier(random_state=0)
        #clf = DecisionTreeClassifier(random_state=0)
        #clf = RandomForestClassifier(random_state=0)
        #clf = ExtraTreesClassifier(random_state=0)
        clf = XGBClassifier(random_state=0)
        classifier = clf.fit(X_train, y_train)
        classifier_arr.append(classifier)

        X_train1 = scaler.transform(X_train0)
        X_fea1 = classifier.predict(X_train1)
        X_fea_list = list(X_fea1)
        data_dict[col_name] = X_fea_list
        print(idx)
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv('train_num'+ str(num) +'.csv',index=False,sep=',')
    print('training end.')


    # 2. array feature of 4 testsets
    save_path1 = 'datatest1.pkl'
    datate = pickle.load(open(save_path1, 'rb'))
    X_pos = datate['X_pos']
    X_neg = datate['X_neg']
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    y_test = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
    y_list = list(y_test)
    test1_dict = {'class':y_list}
    for idx in range(len(data_list)):
        scaler = scaler_arr[idx]
        classifier = classifier_arr[idx]
        y_fea = array_test(datate,scaler,classifier)
        data_name = data_list[idx]
        col_name = data_name[:5]
        test1_dict[col_name] = y_fea
    df_test1 = pd.DataFrame(test1_dict)
    df_test1.to_csv('test1_num'+ str(num) +'.csv',index=False,sep=',')
    print('testing1 end.')

    save_path1 = 'datatest2.pkl'
    datate = pickle.load(open(save_path1, 'rb'))
    X_pos = datate['X_pos']
    X_neg = datate['X_neg']
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    y_test = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
    y_list = list(y_test)
    test2_dict = {'class':y_list}
    for idx in range(len(data_list)):
        scaler = scaler_arr[idx]
        classifier = classifier_arr[idx]
        y_fea = array_test(datate,scaler,classifier)
        data_name = data_list[idx]
        col_name = data_name[:5]
        test2_dict[col_name] = y_fea
    df_test2 = pd.DataFrame(test2_dict)
    df_test2.to_csv('test2_num'+ str(num) +'.csv',index=False,sep=',')
    print('testing2 end.')

    save_path1 = 'datatest3.pkl'
    datate = pickle.load(open(save_path1, 'rb'))
    X_pos = datate['X_pos']
    X_neg = datate['X_neg']
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    y_test = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
    y_list = list(y_test)
    test3_dict = {'class':y_list}
    for idx in range(len(data_list)):
        scaler = scaler_arr[idx]
        classifier = classifier_arr[idx]
        y_fea = array_test(datate,scaler,classifier)
        data_name = data_list[idx]
        col_name = data_name[:5]
        test3_dict[col_name] = y_fea
    df_test3 = pd.DataFrame(test3_dict)
    df_test3.to_csv('test3_num'+ str(num) +'.csv',index=False,sep=',')
    print('testing3 end.')

    save_path1 = 'datatest4.pkl'
    datate = pickle.load(open(save_path1, 'rb'))
    X_pos = datate['X_pos']
    X_neg = datate['X_neg']
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    y_test = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
    y_list = list(y_test)
    test4_dict = {'class':y_list}
    for idx in range(len(data_list)):
        scaler = scaler_arr[idx]
        classifier = classifier_arr[idx]
        y_fea = array_test(datate,scaler,classifier)
        data_name = data_list[idx]
        col_name = data_name[:5]
        test4_dict[col_name] = y_fea
    df_test4 = pd.DataFrame(test4_dict)
    df_test4.to_csv('test4_num'+ str(num) +'.csv',index=False,sep=',')
    print('testing4 end.')
