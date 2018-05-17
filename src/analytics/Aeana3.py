# -*- coding: utf-8 -*-
'''
Created on 2018年5月17日

@author: zwp12
'''





import numpy as np;
import time;
import math;
import os;
from tools import SysCheck
import matplotlib.pyplot as plt 
from autoencoder import Preprocess;

def preprocess(R):
    if R is None:
        return R;
    ind = np.where(R>0);
    newR = np.zeros_like(R);
    newR[ind]=1;
    return  newR; 


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';


us_shape=(339,5825);
# 是否基于用户的自编码器，预测每个用户的所有服务值
isUserAutoEncoder=True;
# 是否基于服务的CF方法
isICF=False;

# 加载AutoEncoder
loadvalues= True;
continue_train = False;
# 加载相似度矩阵
readWcache=False;

axis0 = us_shape[0];
axis1 = us_shape[1];

if isICF:
    axis0 = us_shape[1];
    axis1 = us_shape[0];

mean = 0.908570086101;
ek = 1.9325920405;
# 训练例子
case = 1;
NoneValue = 0.0;

# autoencoder 参数
hidden_node = 150;
learn_rate=0.09;
learn_param = [learn_rate,100,0.99];
repeat = 500;
rou=0.1

# 协同过滤参数
k = 20;
loc_w= 1.0;

test_spa=10;
# 相似列表，shape=(axis0,k),从大到小
S = None;
R = None;

loc_tab=None;

# 相识度矩阵
W = np.full((axis0,axis0), 0, float);
    
fid = 1;
def setFigure(X,Y,fid):
    plt.figure(fid);
    plt.plot(X,Y);
    plt.show();




def encoder_run(spa):
  
    values_path=base_path+'/Dataset/ae_values_space/spa%d'%(spa);
    
    print('开始实验，稀疏度=%d,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    ana_index = np.loadtxt(values_path+'/test_ana_ind.txt',dtype=int);
    ones= np.loadtxt(values_path+'/one_ind.txt',dtype=int);
    zeroes= np.loadtxt(values_path+'/zero_ind.txt',dtype=int);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),len(ana_index)));
    
    print ('转换数据到矩阵开始');
    tnow = time.time();
    ser_ind = np.unique(ana_index[:,1]);
    print(ser_ind);
    print(zeroes);
    print(ones);    
    print(len(ser_ind),len(zeroes),len(ones));
    
    
    zero_err = np.intersect1d(zeroes, ser_ind);
    one_err = np.intersect1d(ones, ser_ind);
    print(zero_err);
    print(one_err);    
    print(len(zero_err),len(one_err));    
    
    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    

    


if __name__ == '__main__':
    spas = [1];
    for spa in spas:
        encoder_run(spa);
    pass