# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:12:35 2021

@author: Administrator
"""

import random

df_num_ratio = 5 #产生终结符列表时dataframe和数字的比例系数（只是系数，不是真实比例）
op_dfnum_ratio = 4 #在产生头部时运算符和终结符的比例系数（只是系数，不是真实比例）
head_len = 6 #头部长度


operator_dict = {'add':2, 'sub':2, 'mul':2, 'div':2, 'sqrt':1,
                 'log':1, 'abso':1, 'sqr':1, 'inv':1, 'maxm':2,
                 'minm':2, 'mean':2, 'delay':2, 'delta':2, 'delta_perc':2,
                 'neg':1, 'scale':1, 'rw_argmax':2, 'rw_argmin':2,
                 'rw_argmaxmin':2, 'rw_max':2, 'rw_min':2, 'rw_ma':2,
                 'rw_std':2, 'rw_maxmin_norm':2, 'rw_rank':2, 'rw_corr':2,
                 'rw_beta':2, 'rw_alpha':2, 'rw_wma':2,
                 'mat_big':2, 'mat_small':2, 'mat_keep':2}
operator_list = list(operator_dict.keys())
data_list = [ 'ClosePrice_fake', 
               'ClosePrice_RecentCt_fake',
               'ClosePrice_SubCt_fake', 'cp_adj_fake', 
                'ForwardSpread', 
               'high_adj_fake',  'low_adj_fake', 'MonthSpread',
                'volume']
num_list = [i for i in range(1, 16)]
ending_list = df_num_ratio * data_list + num_list


def substitute(g):
    if g in operator_list:
        return random.choice(operator_list)
    elif g in data_list:
        return random.choice(data_list)
    else:
        return random.choice(num_list)

def mutate(gene, prob = 0.25):
    new_gene = []  
    for g in gene:
        if random.random() < prob:
            new_gene.append(substitute(g))
        else:
            new_gene.append(g)
    return new_gene


def recombine(a,b):
    la = len(a)
    lb = len(b)
    a_clip = sorted(random.sample(range(la), 2))
    b_clip = sorted(random.sample(range(lb), 2))
    start_a, end_a = a_clip[0], a_clip[1]
    start_b, end_b = b_clip[0], b_clip[1]
    head_a, inner_a, tail_a = a[:start_a], a[start_a:end_a], a[end_a:]
    head_b, inner_b, tail_b = b[:start_b], b[start_b:end_b], b[end_b:]
    new_a = head_a + inner_b + tail_a
    new_b = head_b + inner_a + tail_b
    return new_a, new_b
    
def decoding_gene(gene_list):
    para_pos = 0
    func_pos_list = []
    i = 0
    while i <= para_pos:
        if gene_list[i] in operator_list:
            para_pos += operator_dict[gene_list[i]]
            if para_pos >= len(gene_list):
                return False
            func_pos_list.append(i)
        i += 1
    if len(func_pos_list) > 0:
        return True
    else:
        return False

    
    
    
    
    
    
    
    
    
    