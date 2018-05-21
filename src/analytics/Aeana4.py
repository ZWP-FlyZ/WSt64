# -*- coding: utf-8 -*-
'''
Created on 2018年5月18日

@author: zwp
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
    plt.scatter(X,Y);
    plt.show();




def encoder_run(spa):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';   
    values_path=base_path+'/Dataset/ae_values_space/spa%d'%(spa);
    
    # train_data = test_data;
    print('开始实验，稀疏度=%d,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('转换数据到矩阵开始');
    tnow = time.time();
    u = trdata[:,0];
    s = trdata[:,1];
    u = np.array(u,int);
    s = np.array(s,int);
    R = np.full(us_shape, NoneValue, float);
    R[u,s]=trdata[:,2];
    del trdata,u,s;
    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    print ('预处理数据开始');
    tnow = time.time();
    Preprocess.removeNoneValue(R);
    Preprocess.preprocess(R);
    R=preprocess(R); 
    N = np.count_nonzero(R)
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    if isUserAutoEncoder:
        x_list = np.arange(us_shape[1]);
        sum_list = np.sum(R,axis=0);
    else:
        x_list = np.arange(us_shape[0]);
        sum_list = np.sum(R,axis=1);

    Ps = (sum_list+1)/(N+2);
    Ps = np.log(Ps);
    qsum = np.sum(R * Ps,axis=1);
    # print(qsum);
    NX = np.count_nonzero(R, axis=1);
    print(NX);
    # Q = -1.0 / np.sqrt(NX) * qsum;
    Q = -1.0  * qsum;
    print(Q);
    print(np.sort(Q));
    print(np.median(sum_list),np.mean(sum_list),np.std(sum_list))

    zeros = np.array(np.where(sum_list==0)[0]);
    one = np.array(np.where(sum_list==1)[0]);
#     np.savetxt(values_path+'/zero_ind.txt',zeros,'%d');
#     np.savetxt(values_path+'/one_ind.txt',one,'%d');     
    print(len(zeros));
    print(len(one));
    print(len(np.where(sum_list==2)[0]));
    print(len(np.where(sum_list==3)[0]));
    print(len(np.where(sum_list==4)[0]));
    setFigure(x_list, Ps, spa);

    


if __name__ == '__main__':
    spas = [1];
    for spa in spas:
        encoder_run(spa);
    pass