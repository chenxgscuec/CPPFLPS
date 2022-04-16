# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:25:14 2021

@author: sxg
"""

from Bio import SeqIO
import Bio.SeqIO as Seq

import ORF_length as ORF_len

train_pcts = "Human.coding_RNA_training.fa"

seq_file = train_pcts
select_seq = []

i = 0
for seq in Seq.parse(seq_file,'fasta'):
    Len,Cov,inte_fe = ORF_len.len_cov(seq.seq)
    if(Len < 303):
        select_seq.append(seq)
    i = i + 1
    if(i % 1000 == 0):
        print(i)
filename = 'coding_RNA_sORF.fa'        
Seq.write(select_seq,filename,'fasta')        