# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:40:33 2021

@author: Administrator
"""

import os
import pandas as pd

factor_root_path = "D:\\czy\\specific_kinds\\"
#factor_root_path = "D:\\czy\\daily_factor\\"



file_list = []
for root, dirs, files in os.walk(factor_root_path):
    for file in files:
        file_list.append(os.path.join(root,file))

calma = pd.DataFrame()


for f in file_list:
    df = pd.read_csv(factor_root_path + f[22:])
    df['para'] = f[22:]
    df = df[df['calma']>7]
    df = df[df['calma']<10]
    calma = pd.concat([calma,df])




'''
for f in file_list:
    sep = f.split('\\')
    df = pd.read_csv(factor_root_path + sep[-2] + '\\' + sep[-1])
    df['para'] = sep[-2]
    df = df[df['calma']>7]
    df = df[df['calma']<10]
    calma = pd.concat([calma,df])
'''




#quchongfu!!!




#calma_4.index = range(len(calma_4))
#calma_4.to_csv("D:\\czy\\calma_4.csv")    
    