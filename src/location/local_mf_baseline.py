# -*- coding: utf-8 -*-
'''
Created on 2018年9月5日

@author: zwp12
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;

from mf import MFS;
from location import localtools

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';


NoneValue= 0;
# 初始化参数中 正态分布标准差
rou = 0.1;
# 在矩阵分解中 正则化 参数
lamda = 0.05;

# 隐属性数
f = 100;

#训练次数
repeat = 2
# 学习速率
learn_rate = 0.03;

epoch=100;

spas=[5]

us_shape=(339,5825);
case = 3;
loadvalues=False;
continue_train=True;



def preprocess(train_sets):
    k = len(train_sets);
    ret=[0.0]*k;
    for i in range(k):
        ret[i] = np.mean(train_sets[i][:,2]);
    return ret;

def mf_base_run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
       
    values_path=base_path+'/Dataset/local_mf_baseline_values/spa%.1f_case%d'%(spa,case);
    loc_classes = base_path+'/Dataset/ws/ws_classif_out.txt';
    
    print('开始实验，稀疏度=%.1f,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);
    ser_class = localtools.load_classif(loc_classes);
    classiy_size = len(ser_class);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    
    print ('加载测试数据开始');
    tnow = time.time();
    ttrdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(ttrdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));
    
    print ('分类数据集开始');
    tnow = time.time();
    train_sets = localtools.data_split_class(ser_class, trdata);
    test_sets = localtools.data_split_class(ser_class, ttrdata);
    del trdata,ttrdata;
    print ('分类数据集结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    print ('预处理数据开始');
    tnow = time.time();
    means=preprocess(train_sets);
    # R = R/20.0
    # print(mean,Iu_num,len(Iu_num));
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    ttn = tnow;
    svdes = [MFS.MF_bl_loc(us_shape,f,means[i]) for i in range(classiy_size)];

    if loadvalues:
        for i in range(classiy_size):
            vpp = values_path+'/class%d'%(i);
            svdes[i].preloadValues(vpp);
    if continue_train:
        
        for ep in range(epoch):
            for i in range(classiy_size):
                print ('类%d训练开始'%(i));
                svdes[i].train_mat(train_sets[i], repeat,learn_rate,lamda,values_path);
                vpp = values_path+'/class%d'%(i);
                svdes[i].saveValues(vpp); 
                print ('类%d训练结束，耗时 %.2f秒  \n'%(i,(time.time() - ttn)));
                ttn = time.time();  
            mae=0.0;rmse=0.0;cot=0;
            for i in range(classiy_size):
                for tc in test_sets[i]:
                    if tc[2]<=0:
                        continue;
                    u = int(tc[0]);
                    s = int(tc[1]);
                    rt = svdes[i].predict(u,s);
                    t =abs(rt-tc[2]);
            
                    mae+=t;
                    rmse+=(rt-tc[2])**2;
                    cot+=1;
           
            mae = mae * 1.0 / cot;
            rmse= np.sqrt(rmse/cot);
            print ('-------->>>>ep=%d训练结束，mae=%f耗时 %.2f秒  \n'%(ep,mae,(time.time() - ttn)));
                     
    print ('训练模型结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
    for i in range(classiy_size):
        for tc in test_sets[i]:
            if tc[2]<=0:
                continue;
            u = int(tc[0]);
            s = int(tc[1]);
            rt = svdes[i].predict(u,s);
            t =abs(rt-tc[2]);
    
            mae+=t;
            rmse+=(rt-tc[2])**2;
            cot+=1;
   
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%.1f,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


if __name__ == '__main__':
    for spa in spas:
        for ca in range(1,2):
            case = ca;
            mf_base_run(spa,case);


if __name__ == '__main__':
    pass