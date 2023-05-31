# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:37:38 2021

@author: sxg
"""

import os,sys
import CTD
import ProtParam as PP
import ORF_length as ORF_len
import Bio.SeqIO as Seq
import fickett
import FrameKmer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
#import time
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,confusion_matrix,roc_curve,auc,matthews_corrcoef,precision_recall_fscore_support
#import random
from sklearn.preprocessing import StandardScaler

train_lncRNA = "Homo38.ncrna_training.fa"
train_pcts = "Human.coding_RNA_training.fa"

test_1_lncRNA = "Homo38_ncrna_test.fa"
test_1_pcts = "Human_coding_RNA_test.fa"
test_2_lncRNA = "Homo38.small_ncrna_test.fa"
test_2_pcts = "Human.small_coding_RNA_test.fa"
test_3_lncRNA = "Mouse_ncrna.fa"
test_3_pcts = "Mouse_coding_RNA.fa"
test_4_lncRNA = "Mouse_small_ncrna.fa"
test_4_pcts = "Mouse_small_coding_RNA.fa"

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

def coding_nocoding_potential(input_file):
	coding={}
	noncoding={}
	for line in open(input_file).readlines():
		fields = line.split()
		if fields[0] == 'hexamer':continue
		coding[fields[0]] = float(fields[1])
		noncoding[fields[0]] =  float(fields[2])
	return coding,noncoding

def output_feature(seq_file,species="Human"):
	hex_file = 'Human_Hexamer.tsv'
	coding,noncoding = coding_nocoding_potential(hex_file)
	feature = []     
	length = 0    
	for seq in Seq.parse(seq_file,'fasta'):
        #seq -> seq.seq
		A,T,G,C,AT,AG,AC,TG,TC,GC,A0,A1,A2,A3,A4,T0,T1,T2,T3,T4,G0,G1,G2,G3,G4,C0,C1,C2,C3,C4 = CTD.CTD(seq.seq)
		insta_fe,PI_fe,gra_fe = PP.param(seq.seq)
		fickett_fe = fickett.fickett_value(seq.seq)
		hexamer = FrameKmer.kmer_ratio(seq.seq,6,3,coding,noncoding)
		Len,Cov,inte_fe = ORF_len.len_cov(seq.seq)
		vector = [Len,Cov,inte_fe,hexamer,fickett_fe,insta_fe,PI_fe,gra_fe,A,T,G,C,AT,AG,AC,TG,TC,GC,A0,A1,A2,A3,A4,T0,T1,T2,T3,T4,G0,G1,G2,G3,G4,C0,C1,C2,C3,C4]# + psedncfea
		feature.append(vector)  
		length = length + 1   
		if(length % 2000 == 0):
			print(length)        
	return feature,length         

def save_results(results_file,metrics,time):
    with open(results_file,"a+") as op:
        if os.path.getsize(results_file):
            op.write(str(metrics[0])+"\t"+str(metrics[1])+"\t"+str(metrics[2])+"\t"+str(metrics[3])+"\t"+str(metrics[4])+"\t"+str(metrics[5])+"\t"+str(metrics[6])+"\t"+str(time)+"\n")
        else:
            op.write("Type\tspecificity\trecall\tprecision\taccuracy\tF1\tauc\tmcc\ttime\n")
            op.write(str(metrics[0])+"\t"+str(metrics[1])+"\t"+str(metrics[2])+"\t"+str(metrics[3])+"\t"+str(metrics[4])+"\t"+str(metrics[5])+"\t"+str(metrics[6])+"\t"+str(time)+"\n")

X_pos,len1 = output_feature(train_pcts)
X_neg,len2 = output_feature(train_lncRNA)

data = {'X_pos':X_pos, 'X_neg':X_neg}
save_path = 'datatrain.pkl'
pickle.dump(data,open(save_path, 'wb'))

X_pos,len1 = output_feature(test_1_pcts)
X_neg,len2 = output_feature(test_1_lncRNA)

data = {'X_pos':X_pos, 'X_neg':X_neg}
save_path = 'datatest1.pkl'
pickle.dump(data,open(save_path, 'wb'))

X_pos,len1 = output_feature(test_2_pcts)
X_neg,len2 = output_feature(test_2_lncRNA)

data = {'X_pos':X_pos, 'X_neg':X_neg}
save_path = 'datatest2.pkl'
pickle.dump(data,open(save_path, 'wb'))


X_pos,len1 = output_feature(test_3_pcts)
X_neg,len2 = output_feature(test_3_lncRNA)

data = {'X_pos':X_pos, 'X_neg':X_neg}
save_path = 'datatest3.pkl'
pickle.dump(data,open(save_path, 'wb'))


X_pos,len1 = output_feature(test_4_pcts)
X_neg,len2 = output_feature(test_4_lncRNA)

data = {'X_pos':X_pos, 'X_neg':X_neg}
save_path = 'datatest4.pkl'
pickle.dump(data,open(save_path, 'wb'))

#train_da_list = []
data_list = ['da1p1.pkl','da1p2.pkl','da1p3.pkl','da1p4.pkl','da1p5.pkl','da1p6.pkl',
             'da2p1.pkl','da2p2.pkl','da2p3.pkl','da2p4.pkl','da2p5.pkl','da2p6.pkl',
             'da3p1.pkl','da3p2.pkl','da3p3.pkl','da3p4.pkl','da3p5.pkl','da3p6.pkl',
             'da4p1.pkl','da4p2.pkl','da4p3.pkl','da4p4.pkl','da4p5.pkl','da4p6.pkl',
             'da5p1.pkl','da5p2.pkl','da5p3.pkl','da5p4.pkl','da5p5.pkl','da5p6.pkl',
             'da6p1.pkl','da6p2.pkl','da6p3.pkl','da6p4.pkl','da6p5.pkl','da6p6.pkl']
sorf_file = 'coding_RNA_sORF.fa'
for p_id in range(len(data_list)):
    print(p_id)
    str_mid = data_list[p_id]
    train_da1 = sorf_file[:15]+'_'+ str_mid[:5] +'.fa'
    X_pos,len1 = output_feature(train_da1)
    data = {'X_pos':X_pos}
    save_path = str_mid
    pickle.dump(data,open(save_path, 'wb'))