# -*- coding: utf-8 -*-
'''
Created on 2018年3月19日

@author: zwp
'''


import numpy as np;
import time;
from tools import SysCheck
from AEwithLocation import BPAE,Location,LocBPAE;


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
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j]<0.0:
                R[i,j] = 0.0;    
    #return  (R -  mean) / ek;
    return  R / 20.0; 


# CF预测函数 根据W和S预测出u,s的值,
def predict(u,s,R,W,S):
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
        sum+= W[a0,item]*R[item,a1];
        cot+=W[a0,item];
    if cot != 0:
        return sum/cot;
    else:
        return 0.2;


base_path = r'E:';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';


us_shape=(339,5825);
# 是否基于用户的自编码器，预测每个用户的所有服务值
isUserAutoEncoder=True;
# 是否基于服务的CF方法
isICF=False;

# 加载AutoEncoder
loadvalues= False;
continue_train = True;
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
learn_rate=0.08;
repeat = 500;
rou=0.1
test_spa=20;
# 协同过滤参数
k = 13;


# 相似列表，shape=(axis0,k),从大到小
S = None;
R = None;
# 相识度矩阵
W = np.full((axis0,axis0), 0, float);
    

def encoder_run(spa):
    train_data = base_path+'/Dataset/ws/train/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';   
    values_path=base_path+'/Dataset/ae_values/spa%d'%(spa);
    
    if isUserAutoEncoder:
        loc_path+='/user_info.txt';
    else:
        loc_path+='/ws_info.txt';
    
    
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
    
    print ('地域信息加载开始');
    tnow = time.time();
    lp =  Location.LocationProcesser(loc_path);
    print ('地域信息加载结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('选取特定地域数据开始');
    tnow = time.time();
#     fR = R.copy();
#     loc_name = 'United States';
#     loc_index = lp.loc_dict[loc_name];
#     loc_index = np.array(loc_index)-1;
#     if isUserAutoEncoder:
#         R = R[loc_index,:];
#     else:
#         R = R[:,loc_index];
    print ('选取特定地域数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));     
    
    
    print ('预处理数据开始');
    tnow = time.time();
    R=preprocess(R);
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    print ('选取特定地域数据开始');
    tnow = time.time();
    lae = LocBPAE.LocAutoEncoder(lp,40,R,hidden_node,
                                 [actfunc1,deactfunc1,
                                   actfunc1,deactfunc1],isUserAutoEncoder);
    
    loc_name = 'other2';
    loc_index = lae.loc_aes[loc_name][0];
    loc_index = np.array(loc_index)-1;
    if isUserAutoEncoder:
        R = R[loc_index,:];
    else:
        R = R[:,loc_index];    
    print ('选取特定地域数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    
    
    print ('训练模型开始');
    tnow = time.time();
#     tx = us_shape[0];
#     if isUserAutoEncoder:
#         tx = us_shape[1];
#     encoder = BPAE.BPAutoEncoder(tx,hidden_node,
#                             actfunc1,deactfunc1,
#                              actfunc1,deactfunc1,check_none);
    encoder = lae.loc_aes[loc_name][1];
    if not isUserAutoEncoder:
        R = R.T;
    if loadvalues and encoder.exisValues(values_path):
        encoder.preloadValues(values_path);
    if continue_train:
        encoder.train(R, (learn_rate,100,0.99), repeat,None);
        encoder.saveValues(values_path);
    PR = encoder.calFill(R);####
    if not isUserAutoEncoder:
        R = R.T;
        PR = PR.T;
    
    print(R);
    print();
    print(PR);
    print();
############# PR 还原处理   ###############
    PR = PR * 20.0;
    R = R * 20.0;
    for i in range(PR.shape[0]):
        for j in range(PR.shape[1]):
            if R[i,j]!=NoneValue:
                PR[i,j]=R[i,j];
    print(PR);
    
############# PR 还原处理   ###############        
#     if isUserAutoEncoder:
#         PR = PR.T;
    print ('训练模型开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  


    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
    for tc in trdata:
        uid = int(tc[0]);
        sid = int(tc[1]);
        if tc[2]<=0:
            continue;
        
        if isUserAutoEncoder:
            tagid = uid;
        else:
            tagid = sid;
        
        tids  = np.argwhere(loc_index==tagid);
        if len(tids)==0: continue;
        tid = tids[0,0];
        if isUserAutoEncoder:
            usid = (tid,sid);
        else:
            usid = (uid,tid);
                    
        rt = PR[usid];
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


    print(W)
    print(S)
        
if __name__ == '__main__':
    encoder_run(test_spa);
    pass