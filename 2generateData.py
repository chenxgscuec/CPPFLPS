# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:20:54 2021

@author: sxg
"""
import random
from Bio import SeqIO
import Bio.SeqIO as Seq

seed = 0
max_times = 50
sorf_file = 'coding_RNA_sORF.fa'
p_arr = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]   
len_arr = [10, 20, 30, 40, 50, 60]

#1. mutation
def convert_char_mutation(seq_one,prob):
    len_seq = len(seq_one)
    char = ['A','G','C','T']
#    prob = 0.1
    seq_new = ''
    for idx in range(len_seq):
    #    print(idx)
        p = random.random()
        ch_new = random.choice(char)
        ch_old = seq_one[idx]
        if p <= prob:
            seq_new += ch_new
        else:
            seq_new += ch_old
    return seq_new


# 2. insert
def convert_char_insert(seq_one,prob):
    len_seq = len(seq_one)
#    prob = 0.01
    char = ['A','G','C','T']
    seq_new = ''
    for idx in range(len_seq):
    #    print(idx)
        p = random.random()
        ch_new = random.choice(char)
        ch_old = seq_one[idx]
        if p <= prob:
            ch_insert = ch_new + ch_old
            seq_new += ch_insert
        else:
            seq_new += ch_old    
    return seq_new


# 3. delete
def convert_char_delete(seq_one,prob):
    len_seq = len(seq_one)
#    prob = 0.01
    seq_new = ''
    for idx in range(len_seq):
    #    print(idx)
        p = random.random()
        ch_new = ''
        ch_old = seq_one[idx]
        if p <= prob:
            seq_new += ch_new
        else:
            seq_new += ch_old
    return seq_new


# 4. nucleotide fragment reversal
def convert_str_reversal(seq_one,max_len):
    len_seq = len(seq_one)
    substr_len = random.randint(2,max_len)
    substr_pos = random.randint(0,len_seq-substr_len)
    
    substr_front = seq_one[:substr_pos]
    substr_re = seq_one[substr_pos:substr_pos+substr_len]
    substr_reversion = substr_re[::-1]
    substr_end = seq_one[substr_pos+substr_len:]
    seq_new = substr_front + substr_reversion + substr_end
    return seq_new


# 5. Nucleotide tandem repeat
def convert_str_repeat(seq_one,max_len):
## select a length of substr, select a position, 
## repeat the substr n times, n is a random number in 2-50.
    len_seq = len(seq_one)
    substr_len = random.randint(2,max_len)
    substr_pos = random.randint(0,len_seq-substr_len)
    repeat_times = random.randint(2,max_times)
    
    substr_front = seq_one[:substr_pos]
    substr_re = seq_one[substr_pos:substr_pos+substr_len]
    substr_repeat = substr_re*repeat_times
    substr_end = seq_one[substr_pos+substr_len:]
    seq_new = substr_front + substr_repeat + substr_end
    return seq_new


# 6. delete nucleotide fragments
def convert_str_delete(seq_one,max_len):
    len_seq = len(seq_one)
    substr_len = random.randint(2,max_len)
    substr_pos = random.randint(0,len_seq-substr_len)
    
    substr_front = seq_one[:substr_pos]
    substr_end = seq_one[substr_pos+substr_len:]
    seq_new = substr_front + substr_end
    return seq_new


for p_id in range(len(p_arr)):
    random.seed(seed)
    select_seq = []
    i = 0
    for p_num in range(40):
        for seq in Seq.parse(sorf_file,'fasta'):
            seq_new = seq
            old_data = seq.seq._data
            new_data = convert_char_mutation(old_data,p_arr[p_id])     #1
            seq_new.seq._data = new_data
            select_seq.append(seq_new)
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    filename = sorf_file[:15]+'_da1p'+str(p_id+1)+'.fa'      
    Seq.write(select_seq,filename,'fasta')
    
   
for p_id in range(len(p_arr)):
    random.seed(seed)
    select_seq = []
    i = 0
    for p_num in range(40):
        for seq in Seq.parse(sorf_file,'fasta'):
            seq_new = seq
            old_data = seq.seq._data
            new_data = convert_char_insert(old_data,p_arr[p_id])       #2
            seq_new.seq._data = new_data
            select_seq.append(seq_new)
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    filename = sorf_file[:15]+'_da2p'+str(p_id+1)+'.fa'      
    Seq.write(select_seq,filename,'fasta')    
    
   
for p_id in range(len(p_arr)):
    random.seed(seed)
    select_seq = []
    i = 0
    for p_num in range(40):
        for seq in Seq.parse(sorf_file,'fasta'):
            seq_new = seq
            old_data = seq.seq._data
            new_data = convert_char_delete(old_data,p_arr[p_id])       #3
            seq_new.seq._data = new_data
            select_seq.append(seq_new)
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    filename = sorf_file[:15]+'_da3p'+str(p_id+1)+'.fa'      
    Seq.write(select_seq,filename,'fasta')  
    

for p_id in range(len(len_arr)):
    random.seed(seed)
    select_seq = []
    i = 0
    for p_num in range(40):
        for seq in Seq.parse(sorf_file,'fasta'):
            seq_new = seq
            old_data = seq.seq._data
            new_data = convert_str_reversal(old_data,len_arr[p_id])   #4
            seq_new.seq._data = new_data
            select_seq.append(seq_new)
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    filename = sorf_file[:15]+'_da4p'+str(p_id+1)+'.fa'      
    Seq.write(select_seq,filename,'fasta')      
    

for p_id in range(len(len_arr)):
    random.seed(seed)
    select_seq = []
    i = 0
    for p_num in range(40):
        for seq in Seq.parse(sorf_file,'fasta'):
            seq_new = seq
            old_data = seq.seq._data
            new_data = convert_str_repeat(old_data,len_arr[p_id])     #5
            seq_new.seq._data = new_data
            select_seq.append(seq_new)
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    filename = sorf_file[:15]+'_da5p'+str(p_id+1)+'.fa'      
    Seq.write(select_seq,filename,'fasta')    


for p_id in range(len(len_arr)):
    random.seed(seed)
    select_seq = []
    i = 0
    for p_num in range(40):
        for seq in Seq.parse(sorf_file,'fasta'):
            seq_new = seq
            old_data = seq.seq._data
            new_data = convert_str_delete(old_data,len_arr[p_id])     #6
            seq_new.seq._data = new_data
            select_seq.append(seq_new)
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    filename = sorf_file[:15]+'_da6p'+str(p_id+1)+'.fa'      
    Seq.write(select_seq,filename,'fasta')
