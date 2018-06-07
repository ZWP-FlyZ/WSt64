# -*- coding: utf-8 -*-
'''
Created on 2018年6月7日

@author: zwp
'''

'''
由MF预填补
类DAE进行半填充


由原数据集和参考矩阵进行相似



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
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)))
def deactfunc1(x):
    return x*(1.0-x);
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


############################### 参数区 开始 ###############################
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

# 训练例子
case = 1;
NoneValue = 0.0;

# autoencoder 参数
hidden_node = 150;
learn_rate=0.09
learn_param = [learn_rate,100,0.99];
repeat = 500;
rou=0.1

# 协同过滤参数
k = 17;
loc_w= 1.0;

#矩阵分解特征数
f=100;
#预处理填补比例
cmp_rat=0.12;

# 随机删除比率
cut_rate = 0.5;



# 测试列表
test_spa=[10];
# 地理位置表
loc_tab=None;

############################### 参数区 结束 ###############################

def encoder_run(spa):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
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
    # 填补处理
    Preprocess.preprocessMF_rat(R,mf,isUAE=False,rat=cmp_rat);
    ############################
    
    print(np.sum(R-oriR));
    R/=20.0;# 归一化
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
                             actfunc1,deactfunc1);
    if not isUserAutoEncoder:
        R = R.T;
        oriR =oriR.T;
    if loadvalues and encoder.exisValues(values_path):
        encoder.preloadValues(values_path);
    if continue_train:
        encoder.train(R, oriR,learn_param, repeat,None);
        encoder.saveValues(values_path);
    
    # R = oriR;
    PR = encoder.calFill(R);
#     print(R);
#     print();
#     print(PR);
#     print();
############# PR 还原处理   ###############
    PR = PR * 20.0;
    R = R * 20;
    oriR=oriR*20;
    PR = np.where(R!=NoneValue,R,PR);
    if not isUserAutoEncoder:
        PR = PR.T;
        R = R.T;
        oriR =oriR.T;    
############# PR 还原处理   ###############        
    print ('训练模型结束，耗时 %.2f秒  \n'%((time.time() - tnow)));


    print ('随机删除开始');
    tnow = time.time();
    Preprocess.random_empty(PR, cut_rate);
    print ('随机删除开始，耗时 %.2f秒  \n'%((time.time() - tnow)));



    ###  oriR 原始US矩阵
    ###  R    经过MF处理的US矩阵
    ###  PR   经过随机删除的US 预测矩阵

    print ('生成原矩阵分析开始');
    tnow = time.time();
    b_s,f_s = us_shape;
    us_ana = [[] for _ in range(b_s)];
    for i in range(b_s-1):
        a = oriR[i,:];
        a_not_none = a!=NoneValue;
        a_is_none = a==NoneValue;
        for j in range(i+1,b_s):
            b = oriR[j,:];
            all_have = (b!=NoneValue) & a_not_none;
            none_have =(b==NoneValue) & a_is_none;
            any_have = np.logical_not(all_have | none_have);
            
#             all_p = np.exp(-1.0*np.count_nonzero(all_have)/f_s);
#             non_p = np.exp(-1.0*np.count_nonzero(none_have)/f_s);
#             any_p = np.exp(-1.0*np.count_nonzero(any_have)/f_s);
            
#             all_p = np.exp(np.count_nonzero(all_have)/f_s);
#             non_p = np.exp(np.count_nonzero(none_have)/f_s);
#             any_p = np.exp(np.count_nonzero(any_have)/f_s);            
            
            all_p = 1/(np.count_nonzero(all_have)/f_s);
            non_p = 1/(np.count_nonzero(none_have)/f_s);
            any_p = 1/(np.count_nonzero(any_have)/f_s); 
            
                        
            # us_ana[i].append([all_have,none_have,any_have,all_p,non_p,any_p]);
            us_ana[i].append([all_have,none_have,any_have,150.0,30.0,0.001]);
            # print(len(us_ana[i])); 
    print ('生成原矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));




    print ('计算相似度矩阵开始');
    tnow = time.time();
    mf_R = R;
    R=PR;
    
    # U-CF
    bat_size,feat_size = R.shape;
    W = np.zeros((bat_size,bat_size));
    show_step = int(bat_size/100);
    
    if readWcache and os.path.exists(W_path): 
        del W;  
        W = np.loadtxt(W_path, np.float64);
    else:
        for i in range(bat_size-1):
            if i%show_step ==0:
                print('----->step%d'%(i));
            a = R[i,:];
            for j in range(i+1,bat_size):
                b = R[j,:];
                
                log_and = (a!=0) & (b!=0);
                
                # print([i,j]);
                ####################################
                ws = np.zeros_like(a);
                ana_chp= us_ana[i][j-i-1];
                for indexk in range(3):
                    tmp = log_and & ana_chp[indexk];
                    ws+=np.subtract(a,b,out=np.zeros_like(a),where=tmp) \
                        * ana_chp[indexk+3];
                ws=np.sum(ws**2);
                #####################################
#                 ws=0.0;
#                 ana_chp= us_ana[i][j-i-1];
#                 deta = np.subtract(a,b,out=np.zeros_like(a),
#                                    where=log_and)                
#                 for indexk in range(3):
#                     tmp = log_and & ana_chp[indexk];
#                     ws+=np.multiply(deta,ana_chp[indexk+3],out=np.zeros_like(a),where=tmp);
#                 ws=np.sum(ws**2);                    
                ####################################

#                 deta = np.subtract(a,b,out=np.zeros_like(a),
#                                    where=log_and)
#                 ws = np.sum(deta**2);
                                
                ###################################


                W[i,j]=W[j,i]= 1.0/math.exp(np.sqrt(ws/feat_size));
        np.savetxt(W_path,W,'%.30f');                
    print ('计算相似度矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));


    print ('生成相似列表开始');
    tnow = time.time();
    S = np.argsort(-W)[:,0:k];            
    print ('生成相似列表开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));


    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
#     print('oR',oR);
#     print('R',R);
    for tc in trdata:
        if tc[2]<=0:
            continue;
        rt = predict(int(tc[0]),int(tc[1]),R,W,S);
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));







if __name__ == '__main__':
    for spa in test_spa:
        encoder_run(spa);
    pass