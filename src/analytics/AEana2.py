# -*- coding: utf-8 -*-
'''
Created on 2018年5月7日

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
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';   
    values_path=base_path+'/Dataset/ae_values_space/spa%d'%(spa);
    
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
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    r_list = np.reshape(R,(-1,));
    r_list = r_list[np.where(r_list>0)];
    mean = np.mean(r_list);
    std = np.std(r_list);
    print(mean,std);
    # R = (R-mean)/std;
    delta = mean/std;
    step_range=1000;
    step = 20.0 / step_range;
    boxes = np.zeros((step_range,),float);
    for u in range(us_shape[0]):
        for s in range(us_shape[1]):
            rt = R[u,s];
            if rt==0.0:continue;
            bid = int(rt/step);
            boxes[bid]+=1;
    
    x_list= np.arange(20,step=step);
    

    setFigure(x_list, boxes, spa);

    
    


if __name__ == '__main__':
    spas = [15];
    for spa in spas:
        encoder_run(spa);
    pass