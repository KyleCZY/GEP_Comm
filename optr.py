# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:34:25 2021

"""

import pandas as pd
import numpy as np
import math
EPS_F64 = 1.e-12
#index为Date或Hours

default_time_interval = 10
    
def change_time_interval(x):
    global default_time_interval
    default_time_interval = x
    return None

#rolling函数
def mat_rolling(A, period):  # 针对2D arr, place rolling at z-axis quick than at x-axis
    B = np.concatenate([np.full((period-1,) + A.shape[1:], np.nan), A], axis=0)
    shape = A.shape + (period,)
    strides = B.strides + (B.strides[0],)
    return np.lib.stride_tricks.as_strided(B, shape=shape, strides=strides)

#加法
def add(a, b):
    return a + b

#减法
def sub(a, b):
    return a - b

#乘法
def mul(a, b):
    return a * b

#除法
def div(a, b): 
    if not isinstance(b, pd.DataFrame):
        if b == 0:
            return -1
        else: 
            return a / b
    else:
        return a / b

#开方，有负数则不开
def sqrt(a):
    if isinstance(a, pd.DataFrame):
        if (a.fillna(1) < 0).values.sum() > 0:
            return a
        else:
            return a.apply(lambda x: x ** 0.5)
    elif a <= 0:
        return a
    else:
        return a ** 0.5

#取对数，有负数则不取
def log(a):
    if isinstance(a, pd.DataFrame):
        if (a.fillna(1) <= 0).values.sum() > 0:
            return a
        else:
            return a.applymap(lambda x: math.log(x))
    elif a <= 0:
        return a
    else:
        return math.log(a)

#取绝对值
def abso(a):
    return abs(a)

#平方
def sqr(a):
    return a * a

#倒数，有0则不取
def inv(a):
    if isinstance(a, pd.DataFrame):
        if (a.fillna(1) == 0).values.sum() > 0:
            return a
        else:
            return a.applymap(lambda x: 1/x)
    elif a == 0:
        return a
    else:
        return 1 / a

#最大值    
def maxm(a, b):
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        a_ = (a-b).applymap(lambda x: 1 if x >= 0 else 0)
        b_ = (a-b).applymap(lambda x: 1 if x < 0 else 0)
        return a_ * a + b_ * b
    else:
        return max(a, b)

#最小值 
def minm(a, b):
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        a_ = (a-b).applymap(lambda x: 1 if x < 0 else 0)
        b_ = (a-b).applymap(lambda x: 1 if x >= 0 else 0)
        return a_ * a + b_ * b 
    else:
        return min(a, b)

#均值
def mean(a, b):
    return (a + b) / 2

#a在b个时间之前的值。两个数字则返回NaN，两个矩阵则返回第一个矩阵和10d的结果
def delay(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        return a.shift(default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b>=0:
                return a.shift(int(b))
            else:
                return a.shift(default_time_interval)
        else:
            if a >= 0:
                return b.shift(int(a))
            else:
                return b.shift(default_time_interval)             
    else: #两个数字
        return -1

#a在b个时间长度上的差分，两个数字则返回NaN，两个矩阵则返回第一个矩阵和10d的结果
def delta(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        return a - a.shift(default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 0:
                return (a - a.shift(int(b)))
            else:
                return a - a.shift(default_time_interval)
        else:
            if a >= 0:
                return b - b.shift(int(a))
            else:
                return b - b.shift(default_time_interval)       
    else: #两个数字
        return -1
 
#a在b个时间长度上的变化率，两个数字则返回NaN，两个矩阵则返回第一个矩阵和10d的结果    
def delta_perc(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return (b - b.shift(default_time_interval)) / b
        else:
            return (a - a.shift(default_time_interval)) / a
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 0:
                return (a - a.shift(int(b))) / a
            else:
                return (a - a.shift(default_time_interval)) / a
        else:
            if a >= 0:
                return (b - b.shift(int(a))) / b
            else:
                return (b - b.shift(default_time_interval)) / b            
    else: #两个数字
        return -1

#负数
def neg(a):
    return 0 - a

#截面排序
def rank(a):
    if isinstance(a, pd.DataFrame):
        return a.rank(axis=1)
    else:
        return -1

#截面归一化
def scale(a):
    if isinstance(a, pd.DataFrame):
        return (a.T / a.sum(axis=1)).T
    else:
        return -1

#a过去b个时间内最大值的下标, rolling_window
def rw_argmax(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).apply(lambda x : np.argmax(x))
        else:
            return a.rolling(default_time_interval).apply(lambda x : np.argmax(x))
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).apply(lambda x : np.argmax(x))
            else:
                return a.rolling(default_time_interval).apply(lambda x : np.argmax(x))
        else:
            if a >= 1:
                return b.rolling(int(a)).apply(lambda x : np.argmax(x))
            else:
                return b.rolling(default_time_interval).apply(lambda x : np.argmax(x))  
    else: #两个数字
        return -1

#a过去b个时间内最小值的下标
def rw_argmin(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).apply(lambda x : np.argmin(x))
        else:
            return a.rolling(default_time_interval).apply(lambda x : np.argmin(x))
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).apply(lambda x : np.argmin(x))
            else:
                return a.rolling(default_time_interval).apply(lambda x : np.argmin(x))
        else:
            if a >= 1:
                return b.rolling(int(a)).apply(lambda x : np.argmin(x))
            else:
                return b.rolling(default_time_interval).apply(lambda x : np.argmin(x))  
    else: #两个数字
        return -1

#a过去b个时间内最大最小值的下标之差
def rw_argmaxmin(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).apply(lambda x : np.argmax(x)) - b.rolling(default_time_interval).apply(lambda x : np.argmin(x))
        else:
            return a.rolling(default_time_interval).apply(lambda x : np.argmax(x)) - a.rolling(default_time_interval).apply(lambda x : np.argmin(x))
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).apply(lambda x: np.argmax(x)) - a.rolling(int(b)).apply(lambda x : np.argmin(x))
            else:
                return a.rolling(default_time_interval).apply(lambda x: np.argmax(x)) - a.rolling(default_time_interval).apply(lambda x : np.argmin(x))
        else:
            if a >= 1:
                return b.rolling(int(a)).apply(lambda x: np.argmax(x)) - b.rolling(int(a)).apply(lambda x : np.argmin(x))
            else:
                return b.rolling(default_time_interval).apply(lambda x: np.argmax(x)) - b.rolling(default_time_interval).apply(lambda x : np.argmin(x))  
    else: #两个数字
        return -1   

#a过去b个时间内最大值
def rw_max(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).apply(lambda x: max(x))
        else:
            return a.rolling(default_time_interval).apply(lambda x: max(x))
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).apply(lambda x: max(x))
            else:
                return a.rolling(default_time_interval).apply(lambda x: max(x))
        else:
            if a >= 1:
                return b.rolling(int(a)).apply(lambda x: max(x))
            else:
                return b.rolling(default_time_interval).apply(lambda x: max(x))
    else: #两个数字
        return -1  
    
#a过去b个时间内最小值
def rw_min(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).apply(lambda x: min(x))
        else:
            return a.rolling(default_time_interval).apply(lambda x: min(x))
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).apply(lambda x: min(x))
            else:
                return a.rolling(default_time_interval).apply(lambda x: min(x))
        else:
            if a >= 1:
                return b.rolling(int(a)).apply(lambda x: min(x))
            else:
                return b.rolling(default_time_interval).apply(lambda x: min(x))
    else: #两个数字
        return -1  

#a过去b个时间内均值
def rw_ma(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).mean()
        else:
            return a.rolling(default_time_interval).mean()
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).mean()
            else:
                return a.rolling(default_time_interval).mean()
        else:
            if a >= 1:
                return b.rolling(int(a)).mean()
            else:
                return b.rolling(default_time_interval).mean()
    else: #两个数字
        return -1  

#a过去b个时间内标准差
def rw_std(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return b.rolling(default_time_interval).std()
        else:
            return a.rolling(default_time_interval).std()
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 1:
                return a.rolling(int(b)).std()
            else:
                return a.rolling(default_time_interval).std()
        else:
            if a >= 1:
                return b.rolling(int(a)).std()
            else:
                return b.rolling(default_time_interval).std()
    else: #两个数字
        return -1  

#maxmin标准化
def rw_maxmin_norm(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return (b - b.rolling(default_time_interval).min()) / (b.rolling(default_time_interval).max() - b.rolling(default_time_interval).min())
        else:
            return (a - a.rolling(default_time_interval).min()) / (a.rolling(default_time_interval).max() - a.rolling(default_time_interval).min())
    elif a_type or b_type: #1数字1df
        if a_type:
            a_ = np.array(a.fillna(1))
            a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
            if set(a__) == {0.0, 1.0}:
                return -1
            else:
                if b >= 1:
                    return (a - a.rolling(int(b)).min()) / (a.rolling(int(b)).max() - a.rolling(int(b)).min())
                else:
                    return (a - a.rolling(default_time_interval).min()) / (a.rolling(default_time_interval).max() - a.rolling(default_time_interval).min())
        else:
            b_ = np.array(b.fillna(1))
            b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
            if set(b__) == {0.0, 1.0}:
                return -1
            else:
                if a >= 1:
                    return (b - b.rolling(int(a)).min()) / (b.rolling(int(a)).max() - b.rolling(int(a)).min())
                else:
                    return (b - b.rolling(default_time_interval).min()) / (b.rolling(default_time_interval).max() - b.rolling(default_time_interval).min())  
    else: #两个数字
        return -1 

#a目前取值在过去b时间的排名
def rw_rank(a,b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} and set(b__) == {0.0, 1.0}:
            return -1
        elif set(a__) == {0.0, 1.0}:        
            return pd.DataFrame(mat_rolling(b, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1],index = b.index, columns = b.columns)
        else:
            return pd.DataFrame(mat_rolling(a, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1],index = a.index, columns = a.columns)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return pd.DataFrame(mat_rolling(a, int(b)).argsort(axis = -1).argsort(axis=-1)[:,:,-1],index = a.index, columns = a.columns)
            else:
                return pd.DataFrame(mat_rolling(a, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1],index = a.index, columns = a.columns)
        else:
            if a >= 2:
                return pd.DataFrame(mat_rolling(b, int(a)).argsort(axis = -1).argsort(axis=-1)[:,:,-1],index = b.index, columns = b.columns)
            else:
                return pd.DataFrame(mat_rolling(b, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1],index = b.index, columns = b.columns)
    else: #两个数字
        return -1 
    
#相关系数
def rw_corr(x, y):
    x_type = isinstance(x, pd.DataFrame)
    y_type = isinstance(y, pd.DataFrame)
    if x_type and y_type: #两个df
        x_ = np.array(x.fillna(1))
        x__ = x_.reshape(x_.shape[0]*x_.shape[1],)
        y_ = np.array(y.fillna(1))
        y__ = y_.reshape(y_.shape[0]*y_.shape[1],)
        if set(x__) == {0.0, 1.0} or set(y__) == {0.0, 1.0}:
            return -1
        else:
            x_mean = 1.0 * rw_ma(x, default_time_interval)
            y_mean = 1.0 * rw_ma(y, default_time_interval)
            xx_mean = rw_ma(x * x, default_time_interval)
            xy_mean = rw_ma(x * y, default_time_interval)
            yy_mean = rw_ma(y * y, default_time_interval)
        
            p = xy_mean - x_mean * y_mean
            q1 = xx_mean - x_mean * x_mean
            q2 = yy_mean - y_mean * y_mean
            q1[q1 <= 0.0] = EPS_F64
            q2[q2 <= 0.0] = EPS_F64
            return p / np.sqrt(q1 * q2)
    else:
        return -1
        
#回归的贝塔系数
def rw_beta(x, y):
    x_type = isinstance(x, pd.DataFrame)
    y_type = isinstance(y, pd.DataFrame)
    if x_type and y_type: #两个df
        x_ = np.array(x.fillna(1))
        x__ = x_.reshape(x_.shape[0]*x_.shape[1],)
        y_ = np.array(y.fillna(1))
        y__ = y_.reshape(y_.shape[0]*y_.shape[1],)
        if set(x__) == {0.0, 1.0} or set(y__) == {0.0, 1.0}:
            return -1
        else:
            x_mean = 1.0 * rw_ma(x, default_time_interval)
            y_mean = 1.0 * rw_ma(y, default_time_interval)
            xx_mean = rw_ma(x * x, default_time_interval)
            xy_mean = rw_ma(x * y, default_time_interval)
            p = xy_mean - x_mean * y_mean
            q = xx_mean - x_mean * x_mean
            q[q <= 0.0] = EPS_F64
            return p / q
    else:
        return -1

#回归的alpha系数
def rw_alpha(x, y):
    x_type = isinstance(x, pd.DataFrame)
    y_type = isinstance(y, pd.DataFrame)
    if x_type and y_type: #两个df
        x_ = np.array(x.fillna(1))
        x__ = x_.reshape(x_.shape[0]*x_.shape[1],)
        y_ = np.array(y.fillna(1))
        y__ = y_.reshape(y_.shape[0]*y_.shape[1],)
        if set(x__) == {0.0, 1.0} or set(y__) == {0.0, 1.0}:
            return -1
        else:
            x_mean = 1.0 * rw_ma(x, default_time_interval)
            y_mean = 1.0 * rw_ma(y, default_time_interval)
            xx_mean = rw_ma(x * x, default_time_interval)
            xy_mean = rw_ma(x * y, default_time_interval)
            p = xy_mean - x_mean * y_mean
            q = xx_mean - x_mean * x_mean
            q[q <= 0.0] = EPS_F64
            beta = p / q
            return y_mean - beta * x_mean
    else:
        return -1

#衰减均值
def rw_wma(x, y):
    x_type = isinstance(x, pd.DataFrame)
    y_type = isinstance(y, pd.DataFrame)
    if x_type and y_type: #两个df
        w = 1.0 * np.arange(1, default_time_interval+1)
        w = w / w.sum()
        return pd.DataFrame((mat_rolling(x, default_time_interval) * w[np.newaxis, np.newaxis, :]).sum(axis=-1),index = x.index, columns = x.columns)
    elif x_type or y_type: #1数字1df
        if x_type:
            if y >= 1:
                w = 1.0 * np.arange(1, int(y)+1)
                w = w / w.sum()
                return pd.DataFrame((mat_rolling(x, int(y)) * w[np.newaxis, np.newaxis, :]).sum(axis=-1),index = x.index, columns = x.columns)
            else:
                w = 1.0 * np.arange(1, default_time_interval+1)
                w = w / w.sum()
                return pd.DataFrame((mat_rolling(x, default_time_interval) * w[np.newaxis, np.newaxis, :]).sum(axis=-1),index = x.index, columns = x.columns)
        else:
            if x >= 1:
                w = 1.0 * np.arange(1, int(x)+1)
                w = w / w.sum()
                return pd.DataFrame((mat_rolling(y, int(x)) * w[np.newaxis, np.newaxis, :]).sum(axis=-1),index = y.index, columns = y.columns)
            else:
                w = 1.0 * np.arange(1, default_time_interval+1)
                w = w / w.sum()
                return pd.DataFrame((mat_rolling(y, default_time_interval) * w[np.newaxis, np.newaxis, :]).sum(axis=-1),index = y.index, columns = y.columns)
    else:
        return -1

#比较两个矩阵大小，返回01矩阵大
def mat_big(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        return (a>b) * 1.0
    else:
        return -1

#比较两个矩阵大小，返回01矩阵小
def mat_small(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        return (a<b) * 1.0
    else:
        return -1

#比较两个矩阵大小，返回较大值
def mat_keep(a, b):
    a_type = isinstance(a, pd.DataFrame)
    b_type = isinstance(b, pd.DataFrame)
    if a_type and b_type: #两个df
        a_ = np.array(a.fillna(1))
        a__ = a_.reshape(a_.shape[0]*a_.shape[1],)
        b_ = np.array(b.fillna(1))
        b__ = b_.reshape(b_.shape[0]*b_.shape[1],)
        if set(a__) == {0.0, 1.0} or set(b__) == {0.0, 1.0}:
            return a * b
        else:
            return -1
    else:
        return -1
           

