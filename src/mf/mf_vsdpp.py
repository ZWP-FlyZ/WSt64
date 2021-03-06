# -*- coding: utf-8 -*-
'''
Created on 2018年2月1日

@author: zwp12
'''

'''
将矩阵分解和基准线预测结合
https://www.cnblogs.com/Xnice/p/4522671.html
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;

from mf import MFS;
    
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
lamda = 0.04;

# 隐属性数
f = 100;

#训练次数
repeat = 150
# 学习速率
learn_rate = 0.03;


us_shape=(339,5825);
case = 1;
loadvalues=False;
continue_train=True;

def mf_base_run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
       
    values_path=base_path+'/Dataset/mf_vsdpp_values/spa%d'%(spa);
    
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
    # print(mean,Iu_num,len(Iu_num));
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    tx = us_shape[0];

    svd = MFS.MF_bl_plus(R.shape,f,mean);

    if loadvalues and svd.exisValues(values_path):
        svd.preloadValues(values_path);
    if continue_train:
        svd.train_mat(R, repeat,learn_rate,lamda,None);
        svd.saveValues(values_path);
                
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
        if tc[2]<=0:
            continue;
        rt = svd.predict(int(tc[0]),int(tc[1]));
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


if __name__ == '__main__':
    for spa in [1,2,3,4,5]:
        mf_base_run(spa,case)
    
    pass