# -*- coding: utf-8 -*-
'''
Created on 2018年5月14日

@author: zwp
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;

from adaboost import MFadaboost;


    
def preprocess(R):
    if R is None:
        return R;
    ind = np.where(R<0);
    R[ind]=0;
    mean = np.sum(R)/np.count_nonzero(R);
    Iu_num = np.count_nonzero(R,axis=1);
    #return  (R -  mean) / ek;
    return  mean,Iu_num;


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';

NoneValue= 0;
# 初始化参数中 正态分布标准差
rou = 0.1;
# 在矩阵分解中 正则化 参数
lamda = [0.04,0.04,0.04,0.04,0.04,0.04];

# 隐属性数
f = 32;

#训练次数
repeat = [140,140,140,140,140,140]
# 学习速率
learn_rate = [0.02,0.02,0.020,0.02,0.02,0.02];

# adaboost迭代次数
K=1;

us_shape=(339,5825);
case = 2;
loadvalues=False;
continue_train=True;

def mf_base_run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
       
    values_path=base_path+'/Dataset/mf_baseline_values/spa%d'%(spa);
    
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
    mean,Iu_num=preprocess(R);
    # R = R/20.0
    # print(mean,Iu_num,len(Iu_num));
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    tx = us_shape[0];

    adamf = MFadaboost.mf_adaboost(R,K);
    if loadvalues:
        adamf.load_param(values_path);
        if continue_train:
            adamf.update_train(f, repeat, learn_rate, lamda);
    else:
        adamf.train(f,repeat,learn_rate,lamda);
    adamf.save_param(values_path);
    print ('训练模型开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  


    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
    ana = np.zeros(us_shape);
    R_ana = np.zeros(us_shape);
    for tc in trdata:
        if tc[2]<=0:
            continue;
        u = int(tc[0]);
        s = int(tc[1]);
        rt = adamf.predict(u,s);
        t =abs(rt-tc[2]);
        ana[u,s]=t;
        R_ana[u,s]=tc[2];
        mae+=t;
        rmse+=(rt-tc[2])**2;
        cot+=1;
    list_ana = ana.reshape((-1,));    
    ind = np.argsort(-list_ana)[:100];
    ana_sorted = list_ana[ind];
    arg_list = [[int(i/us_shape[1]),int(i%us_shape[1])]for i in ind];
    ori_list = [R_ana[i[0],i[1]] for i in arg_list];
    np.savetxt(values_path+'/test_ana_value.txt',np.array(ana_sorted),'%.6f');
    np.savetxt(values_path+'/test_ana_ind.txt',np.array(arg_list),'%d');
    np.savetxt(values_path+'/test_ana_ori_value.txt',np.array(ori_list),'%.6f');

    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


if __name__ == '__main__':
    for spa in [2]:
        mf_base_run(spa,case)
    
    pass
