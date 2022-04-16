CPPFLPS
===============================
An implementation of CPPFLPS, a new model for distinguishing ncRNAs from coding RNAs.

Reference
========================
Our manuscipt titled with "Non-coding RNA identification with pseudo RNA sequences and feature representation learning" is being reviewed.

Requirements
========================
    [python 3.6](https://www.python.org/downloads/)

Usage
========================
if you want to run the coding potential prediction of RNA sequences without data augmentation, you can run:
python 4CPP.py

if you want to run the feature representation learning method, you can run:
python 5FL.py and 6class.py

1SelectData.py is used for selecting RNA sequences of ORF length less than 303nt from coding RNAs.

2generateData.py is used for gererating pseudo RNA sequences for data augmentation.

3Feature.py is uesd for feature calculation.


Data
=====================
In this work, we use the datasets and part code organized by CPPred(http://www.rnabinding.com/CPPred/).




Contact
=====================
Author: Xian-gan Chen
Maintainer: Xian-gan Chen
Mail: chenxg@mail.scuec.edu.cn
Date: 2022-4-8
School of Biomedical Engineering, South-Central Minzu University, China
