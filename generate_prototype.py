# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:56:24 2021

@author: Administrator
"""

import random
import pandas as pd
import numpy as np
import optr as op
import os

path = 'D:\\czy\\Data_daily\\'

df_num_ratio = 5 #产生终结符列表时dataframe和数字的比例系数（只是系数，不是真实比例）
op_dfnum_ratio = 4 #在产生头部时运算符和终结符的比例系数（只是系数，不是真实比例）
head_len = 6 #头部长度
time_interval = 5 #default_time_interval在optr.py中调整
op.change_time_interval(time_interval)

rpath = 'D:\\czy\\daily_factor\\%d,%d,%d,%d\\' % (df_num_ratio, op_dfnum_ratio, head_len, time_interval)
if not os.path.exists(rpath):
    os.makedirs(rpath)


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
yizi = pd.read_csv(path + 'yizi.csv').set_index('Date') #hour
RawRet = pd.read_csv(path + 'RawRet.csv').set_index('Date') #hour
'''
ending_list = ['ClosePrice_fake', 'ClosePrice_RecentCt_fake', 
               'ClosePrice_SubCt_fake', 'cp_adj_fake', 'ForwardSpread', 
               'high_adj_fake', 'low_adj_fake', 'MonthSpread']
num_list = [i for i in range(1, 61)]
ending_list = 40 * data_list + num_list
'''


def generate_gene(head_len):
    max_para = max(operator_dict.values())
    tail_len = head_len * (max_para - 1) + 1
    head_list = [random.choice(operator_list)] + random.choices(op_dfnum_ratio * operator_list + ending_list, k = head_len - 1)
    tail_list = random.choices(ending_list, k = tail_len)
    gene_list = head_list + tail_list
    return gene_list

def ending_to_data(x):
    return pd.read_csv(path + x + '.csv').set_index('Date') #'hour'

def decoding_gene(gene_list):
    para_pos = 0
    func_pos_list = []
    i = 0
    while i <= para_pos:
        if gene_list[i] in operator_list:
            para_pos += operator_dict[gene_list[i]]
            func_pos_list.append(i)
        i += 1
    if len(func_pos_list) > 0:
        raw_list = [ending_to_data(i) if i in data_list else i for i in gene_list[:para_pos + 1]]
        for j in func_pos_list[::-1]:
            if operator_dict[raw_list[j]] == 2:
                exec("raw_list[j] = op.%s(raw_list[-2],raw_list[-1])" % raw_list[j])
                #res_list.append(raw_list[j])
                raw_list = raw_list[:-2]
            else:
                exec("raw_list[j] = op.%s(raw_list[-1])" % raw_list[j])
                #res_list.append(raw_list[j])
                raw_list = raw_list[:-1]
    return gene_list[:para_pos + 1], raw_list[0]
                
#适应度函数
def fitness_func(factor_tmp,yizi,RawRet):
    fee = 0.0004
    hedgenum = 5
    #生成仓位
    factor_long = factor_tmp.rank(axis = 1)    
    factor_short = (-factor_tmp).rank(axis = 1) 
    factor_long = (factor_long<=hedgenum)
    factor_long = factor_long.astype("int")
    factor_short = (factor_short<=hedgenum)
    factor_short = factor_short.astype("int")    
    factor_short = -factor_short
    factor = factor_long + factor_short
    
    #涨跌停或停牌持仓调控
    iftrade = yizi.copy()
    iftrade = iftrade.fillna(0) #nan默认为不在指数内，但是可正常交易
    iftrade = iftrade.replace(1, np.nan)
    iftrade = iftrade.replace(0, 1)
    factor_array = np.multiply(np.array(factor),np.array(iftrade))
    factor = pd.DataFrame(factor_array, index = yizi.index, columns = yizi.columns)
    factor = factor.fillna(method = 'pad').fillna(0)

    #清算所选品种收益和换手
    retmat = RawRet.copy()
    retposmat = np.multiply(np.array(factor.shift()),np.array(retmat))
    retposmat = pd.DataFrame(retposmat, index = retmat.index, columns = retmat.columns)
    fee = factor.diff().abs()*fee
    retposmat = retposmat - fee
    Retdf = retposmat.sum(axis = 1)/hedgenum/2
    Retdf = Retdf.reset_index().rename(columns = {0:'Ret'})
    if 'hour' in Retdf.columns:
        Retdf['Date'] = Retdf['hour'].map(lambda x: x[0:10])
        Retdf = Retdf.groupby('Date')['Ret'].sum().reset_index()
        
    #计算盈亏比
    Retdf['Retcumsum'] = Retdf['Ret'].cumsum()
    Retdf['Retcumsummax'] = Retdf['Retcumsum'].cummax()
    Retdf['dd'] = Retdf['Retcumsum'] - Retdf['Retcumsummax']         
    calma = -Retdf['Ret'].sum()/Retdf['dd'].min()
    return calma , Retdf 
        

population = 3000
repeat = 3

for r in range(repeat):
    print(r)
    eff_list = []
    factor_list = []
    calma_list = []
    for pop in range(population + 1):
        gene_list = generate_gene(head_len)
        #pd.DataFrame({"factor":gene_list}).to_csv(rpath + 'g.csv')
        eff_gene, factor = decoding_gene(gene_list)
        if isinstance(factor, pd.DataFrame):
            f_ = np.array(factor.fillna(1))
            f__ = f_.reshape(f_.shape[0]*f_.shape[1],)
            if set(f__) != {0.0, 1.0} and factor.isnull().sum().sum() < 0.5 * factor.shape[0] * factor.shape[1]:
                factor_list.append(gene_list)
                eff_list.append(eff_gene)
                calma_list.append(fitness_func(factor,yizi,RawRet)[0])
        if pop % 10 == 0:
            print(pop, end = ' ')
    pd.DataFrame({"factor":factor_list, 'calma': calma_list, 'eff_gene':eff_list}).to_csv(rpath + str(r) + '.csv')
    print()
            





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    