# -*- coding: utf-8 -*-
'''
Created on 2018年5月16日

@author: zwp12
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck
from autoencoder import Preprocess;

from autoencoder import BPAE
from tools.LoadLocation import loadLocation
from mf.MFS import MF_bl;

def actfunc1(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)));


def deactfunc1(x):
    return x*(1.0-x);


def actfunc2(x):
    return x;


def deactfunc2(x):
    return 1;


def check_none(x):
    if x is None:
        return True;
    elif x <= 0.0:
        return True;
    return False;


def preprocess(R):
    if R is None:
        return R;
    ind = np.where(R<0);
    R[ind]=0;
    #return  (R -  mean) / ek;
    return  R / 20.0; 


# CF预测函数 根据W和S预测出u,s的值,
def predict(u,s,R,W,S):
    global loc_tab;
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


def predict_for_s(u,s,R,W,S):
    global loc_tab;
    a0 = s;
    a1 = u;
    
    sum = 0.0;cot=0.0;
    for item in S[a0,:]:
        if W[a0,item]<=0.0:
            break;
        if R[item,a1] ==NoneValue:
            continue;
        rw = (W[a0,item]);            
#         if loc_tab[a0]==loc_tab[item]:
#             rw *=loc_w;
        
        sum+= rw*R[item,a1];
        cot+=rw;
    if cot != 0:
        return sum/cot;
    else:
        return 0.2;



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
case = 2;
NoneValue = 0.0;

# autoencoder 参数
hidden_node = 150;
learn_rate=0.09
learn_param = [learn_rate,100,0.99];
repeat = 500;
rou=0.1

cut_rate = 0.5;


# 协同过滤参数
k = 17; # user-cf-k;
sk = 17; # service-cf-k;
cf_w = 0.1;


load_SW =  True;


loc_w= 1.0;

f=100;
cmp_rat=0.2;

test_spa=2;
# 相似列表，shape=(axis0,k),从大到小
S = None;
R = None;

loc_tab=None;

# 相识度矩阵
W = np.full((axis0,axis0), 0, float);
    

def encoder_run(spa):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
    SW_path = base_path+'/Dataset/ws/BP_CF_SW_spa%d_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';   
    values_path=base_path+'/Dataset/dae_values/spa%d'%(spa);
    
    mf_values_path=base_path+'/Dataset/mf_baseline_values/spa%d'%(spa);
    
    
    
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
    oriR = R.copy();
    ############################
    # 矩阵分解填补预处理
    mean = np.sum(R)/np.count_nonzero(R);
    mf = MF_bl(R.shape,f,mean);
    mf.preloadValues(mf_values_path);
    
    
    ############################
    Preprocess.preprocessMF_rat(R,mf,rat=cmp_rat);
    print(np.sum(R-oriR));
    R/=20.0;
    oriR/=20.0;
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
        
    print ('加载地理位置信息开始');
    tnow = time.time();
    if isICF:
        loc_path+='/ws_info.txt';
    else:
        loc_path+='/user_info.txt';
    global loc_tab;        
    loc_tab = loadLocation(loc_path);
    print ('加载地理位置信息完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),len(loc_tab)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    tx = us_shape[0];
    if isUserAutoEncoder:
        tx = us_shape[1];
    encoder = BPAE.DenoiseAutoEncoder(tx,hidden_node,
                            actfunc1,deactfunc1,
                             actfunc1,deactfunc1,check_none);
    if not isUserAutoEncoder:
        R = R.T;
    if loadvalues and encoder.exisValues(values_path):
        encoder.preloadValues(values_path);
    if continue_train:
        encoder.train(R, oriR,learn_param, repeat,None);
        encoder.saveValues(values_path);
    
    # R = oriR;
    PR = encoder.calFill(R);
    print(R);
    print();
    print(PR);
    print();
############# PR 还原处理   ###############
    PR = PR * 20.0;
    R = R * 20;
    oriR=oriR*20;
    PR = np.where(R!=NoneValue,R,PR);
    print(PR);
    if not isUserAutoEncoder:
        PR = PR.T;
        R = R.T;    
############# PR 还原处理   ###############        
    print ('训练模型开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  


    print ('随机删除开始');
    tnow = time.time();
    Preprocess.random_empty(PR, cut_rate);
    print ('随机删除开始，耗时 %.2f秒  \n'%((time.time() - tnow)));



    global W,S;
    print ('计算相似度矩阵开始');
    tnow = time.time();
    oR = R;
    R=PR;
    for i in range(axis0-1):
        if i%50 ==0:
            print('----->step%d'%(i))
        for j in range(i+1,axis0):
            ws = 0.0;
            a = R[i,:];
            b = R[j,:];
            # log = 
            deta = np.subtract(a,b,out=np.zeros_like(a),
                               where=((a!=NoneValue) & (b!=NoneValue)))
            ws += np.sum(deta**2);
            W[i,j]=W[j,i]= 1.0/math.exp(np.sqrt(ws/axis1));

            # origin W[i,j]=W[j,i]=1.0/(ws ** (1.0/p)+1.0);
            # W[i,j]=W[j,i]=1.0/( ((ws/cot) ** (1.0/p))+1.0);
            
            # W[i,j]=W[j,i]= 1.0/math.exp(((ws) ** (1.0/p))/cot);
    np.savetxt(W_path,W,'%.30f');
    
    R=PR.T;
    SW = np.zeros((axis1,axis1));
    
    if os.path.exists(SW_path) and load_SW:
        SW = np.loadtxt(SW_path,np.float64);
    else:
        for i in range(axis1-1):
            if i%50 ==0:
                print('----->step%d'%(i))
            for j in range(i+1,axis1):
                ws = 0.0;
                a = R[i,:];
                b = R[j,:];
                # log = 
                deta = np.subtract(a,b,out=np.zeros_like(a),
                                   where=((a!=NoneValue) & (b!=NoneValue)))
                ws += np.sum(deta**2);
                SW[i,j]=SW[j,i]= 1.0/math.exp(np.sqrt(ws/axis1));
    
                # origin W[i,j]=W[j,i]=1.0/(ws ** (1.0/p)+1.0);
                # W[i,j]=W[j,i]=1.0/( ((ws/cot) ** (1.0/p))+1.0);
                
                # W[i,j]=W[j,i]= 1.0/math.exp(((ws) ** (1.0/p))/cot);
        np.savetxt(SW_path,SW,'%.10f');    
    
    R = PR;
                    
    print ('计算相似度矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));


    print ('生成相似列表开始');
    tnow = time.time();
    S = np.argsort(-W)[:,0:k];
    SS = np.argsort(-SW)[:,0:sk];            
    print ('生成相似列表开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));




    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
    print('oR',oR);
    print('R',R);
    for tc in trdata:
        if tc[2]<=0:
            continue;
        urt = predict(int(tc[0]),int(tc[1]),R,W,S);
        srt = predict_for_s(int(tc[0]),int(tc[1]), R.T, SW, SS)
        rt = cf_w * urt + (1-cf_w)*srt;
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


    print(W)
    # print(S)
        
if __name__ == '__main__':
    spas = [test_spa];
    for spa in spas:
        encoder_run(spa);
    pass


