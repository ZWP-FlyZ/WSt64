# -*- coding: utf-8 -*-
'''
Created on 2018年3月23日

@author: zwp
'''

import numpy as np;
import time;
from tools import SysCheck
from AEwithLocation import Location,LocBPAE,FeatureNN;

def actfunc1(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)));
  
  
def deactfunc1(x):
    return x*(1.0-x);


# def actfunc1(x):
#     return np.log(1+np.exp(x));
#  
#  
# def deactfunc1(x):
#     return 1-np.exp(-x);



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

def getComp(f_size,a):
    '''
    获得补集
    '''
    return np.setdiff1d(np.arange(f_size),np.array(a));



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
learn_rate=0.09;
learn_param = [learn_rate,100,0.99];
repeat = 200;
rou=0.1
test_spa=20;
# 协同过滤参数
k = 13;

oeg = 40;
name_extend_data=['United States'];
k_extend_data=[41];

name_list_train=['p_all'];
name_list_pr=['p_all','all'];

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
    values_path=base_path+'/Dataset/loc_ae_values/spa%d'%(spa);
    
    FNN_path = base_path+'/Dataset/loc_ae_values/fnn';
    
    
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
    
    
    
    print ('预处理数据开始');
    tnow = time.time();
    R=preprocess(R);
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    print ('FNN开始');
    tnow = time.time();
    lae = LocBPAE.LocAutoEncoder(lp,oeg,R,hidden_node,
                             [actfunc1,deactfunc1,
                               actfunc1,deactfunc1],isUserAutoEncoder);
                               
    hnn = FeatureNN.HidNN(FNN_path,
                          isUserAutoEncoder,
                          R,[test_spa,case,NoneValue],
                          );
    f_size = us_shape[0];
    if not isUserAutoEncoder:
        f_size = us_shape[1];
    for i in range(len(name_extend_data)):
        n  = name_extend_data[i];
        nk = k_extend_data[i];
        a = lae.getIndexByLocName(n);
        ca = getComp(f_size, a);
        extend_index = hnn.getExtendDataIndex2(a, ca, nk);
        # extend_index = [69, 70, 78, 79, 83, 84, 85, 86, 97, 98, 99, 116, 117, 118, 119, 132, 133, 142, 143, 148, 149, 150, 151, 152, 153, 167, 168, 174, 175, 176, 177, 235, 236, 261, 262, 263, 264, 284, 285, 328, 329];
        print(n,extend_index);
        lae.extendData(n, extend_index);
        pass;                  
#     ex = lae.getWeightByLocNameWithExt('p_all');
#     print(ex);
    print ('FNN结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    us_index = lae.getIndexByLocName('United States');
    ger_index = lae.getIndexByLocName('Germany');
    
    US = R[us_index,:];
    GER = R[ger_index,:];
    
    print(US);
    print(GER);
    
    
    
    
    
    mae=0.0;
    rmse=0.0;
    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


    print(W)
    print(S)
        
if __name__ == '__main__':
    encoder_run(test_spa);
    pass