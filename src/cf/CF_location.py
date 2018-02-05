# -*- coding: utf-8 -*-
'''
Created on 2018年1月16日

@author: zwp12
'''

import numpy as np;
import time;
import math;
from math import sqrt
import os;
from tools import SysCheck
from tools.LoadLocation import loadLocation

'''
协同过滤算法，在TensorFlow下的实现，
标志isICF如果为true,则是基于物品的cf;否则为基于用户的CF

一下方法将-1值作为空值，而不是一个状态值

'''


base_path = r'E:';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';

readWcache = False;

# 数据输入形状
isICF = False;
us_shape= (142,4532);
us_shape= (339,5825);

NoneValue = -1;

axis0 = us_shape[0];
axis1 = us_shape[1];
if isICF:
    axis0 = us_shape[1];
    axis1 = us_shape[0];
 
# 相识度矩阵
W = np.full((axis0,axis0), 0, float);

# 相似列表，shape=(axis0,k),从大到小
S = None;
R = None;

loc_tab=None;

sumS=np.zeros(axis0,float);# 平均向量

spas = [5,10,15,20] #稀疏度

case = 1;# 训练与测试例
k=100; #
p = 2 #

loc_w = 5;


# 根据W和S预测出u,s的值,
def predict(u,s):
    global R,W,S,sumS,loc_tab;
    a0 = u;
    a1 = s;
    if isICF:
        a0 = s;
        a1 = u;
    sum = 0.0;cot=0.0;
    for item in S[a0,:]:
        if W[a0,item]<=0.0:
            break;
        if R[item,a1] ==NoneValue:
            continue;
        rw = (W[a0,item]);            
        if loc_tab[a0]==loc_tab[item]:
            rw *=loc_w;
        
        sum+= rw*R[item,a1];
        cot+=rw;
    if cot != 0:
        return sum/cot;
    else:
        return 0.2;
    
def run_cf(spa):
    global R,W,S,sumS,loc_tab;
    
    train_data = base_path+'/Dataset/ws/train/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';
    
    print('开始实验，isICF=%s,稀疏度=%d,case=%d'%(isICF,spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));

    print ('加载地理位置信息开始');
    tnow = time.time();
    if isICF:
        loc_path+='/ws_info.txt';
    else:
        loc_path+='/user_info.txt';        
    loc_tab = loadLocation(loc_path);
    n = np.alen(trdata);
    print ('加载地理位置信息完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    
    print ('转换数据到矩阵开始');
    tnow = time.time();
    u = trdata[:,0];
    s = trdata[:,1];
    u = np.array(u,int);
    s = np.array(s,int);
    R = np.full(us_shape, NoneValue, float);
    R[u,s]=trdata[:,2];
    if isICF:
        R = R.T;
    # mean = np.mean(R,1);
    del trdata,u,s;
    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    print ('计算相似度矩阵开始');
    tnow = time.time();
    i=0;
    if readWcache and os.path.exists(W_path):
        W = np.loadtxt(W_path, np.float128);
    else:
        for i in range(axis0-1):
            if i%50 ==0:
                print('----->step%d'%(i))
            for j in range(i+1,axis0):
                ws = 0.0;
                cot = 0;
                for c in range(axis1):
                    if R[i,c]!=NoneValue and R[j,c]!=NoneValue:
                        ws += abs(R[i,c]-R[j,c])**p;
                        cot+=1;
                if cot!= 0:
                    # origin W[i,j]=W[j,i]=1.0/(ws ** (1.0/p)+1.0);
                    # W[i,j]=W[j,i]=1.0/( ((ws/cot) ** (1.0/p))+1.0);
                    # W[i,j]=W[j,i]= 1.0/math.exp((ws/cot) ** (1.0/p));
                    W[i,j]=W[j,i]= 1.0/math.exp((ws) ** (1.0/p));
                    # W[i,j]=W[j,i]= 1.0/math.exp(((ws) ** (1.0/p))/cot);
        np.savetxt(W_path,W,'%.30f');                
    print ('计算相似度矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));


    print ('生成相似列表开始');
    tnow = time.time();
    S = np.argsort(-W)[:,0:k];
    for i in range(axis0):
        sumS[i] = np.sum(W[i,S[i]]);            
    print ('生成相似列表开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
    for tc in trdata:
        if tc[2]<0:
            continue;
        rt = predict(int(tc[0]),int(tc[1]));
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    
    
    print('实验结束，总耗时 %.2f秒,isICF=%s,稀疏度=%d,MAE=%.3f,RMSE=%.3f\n'%((time.time()-now),isICF,spa,mae,rmse));
    print('----------------------------------------------------------\n');


    print(W);
    print(S);    
if __name__ == '__main__':
#     for spa in spas:
#         run_cf(spa);
    run_cf(5);
    pass