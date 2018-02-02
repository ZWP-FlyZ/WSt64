# -*- coding: utf-8 -*-
'''
Created on 2018年1月15日

@author: zwp12
'''


'''

为example1 生成数据集

1.生成 wst_zero,wst_all_mean,wst_us_mean，完整数据集（142*4532*64）
2.按照稀疏度列表生成训练集与测试集，采用剩下数据都是测试集的方法。
3.其中wst_zero为类型1，wst_all_mean为类型2，wst_us_mean为类型3。

'''

import time 
import numpy as np
import random 

example = 'example1';

types_name = [
            'wst_zero',
            'wst_all_mean',
            'wst_us_mean'
        ];

base_path = r'E:/Dataset/wst';
origin_data = base_path+'/rtdata.txt';

#origin_data = r'E:/Dataset/my/mytest2.txt';

dataset_out_path = base_path +'/'+example;

spas = [5,10,15,20,30]; # 稀疏度

ust_shape=(142,4532,64); # 数据形状
#ust_shape=(7,7,7);
NoneValue = -1;

ust_mat = np.full(ust_shape, NoneValue, float);# 数据集
us_mean = np.full((142,4532),NoneValue,float);# us评价矩阵
nv_locs = [];# 空值位置

def randomin(a):
    tmp = random.random()*0.07;
    tmp = a + random.uniform(-tmp,tmp);
    if tmp <0 or tmp>20:
        tmp = 2.0*a-tmp;
    return tmp;   

def split(typename,typeid):
    global spas,ust_mat;
    for spa in spas:
        print ('------->类型： '+typename+' 稀疏度：%d 分割开始 '%(spa));
        snow = time.time();
        train_set=[];test_set=[];
        for i in range(ust_shape[0]):
            for j in range(ust_shape[1]):
                for k in range(ust_shape[2]):
                    if random.randint(1,100) <= spa:
                        train_set.append((i,j,k,ust_mat[i,j,k]));
                    else:
                        test_set.append((i,j,k,ust_mat[i,j,k]));
        cotn = len(train_set);
        train_path = dataset_out_path+r'/train/sparseness%d/train_t%d_%d.txt'%(spa,typeid,1);
        np.savetxt(train_path,np.array(train_set),'%.2f');
        del train_set;
        test_path = dataset_out_path+r'/test/sparseness%d/test_t%d_%d.txt'%(spa,typeid,1);
        np.savetxt(test_path,np.array(test_set),'%.2f');
        del test_set;        
        print ('------->类型： '+typename+
               ' 稀疏度：%d 分割结束，耗时 %.2f秒  train_set_len = %d '%(spa,(time.time() - snow),cotn));
              
def spilter():
    
    print ('加载数据开始');
    now = time.time();
    trdata = np.loadtxt(origin_data, dtype=float);
    n = np.alen(trdata);
    print ('加载数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('转换数据到矩阵开始');
    tnow = time.time();
    tu = np.array(trdata[:,0],dtype=int);
    ts = np.array(trdata[:,1],dtype=int);
    tt = np.array(trdata[:,2],dtype=int);
    tv = np.array(trdata[:,3],dtype=float);
    ust_mat[tu,ts,tt] = tv; 
    del tu,ts,tt,tv,trdata;
    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    

    print ('统计计算开始');
    tnow = time.time();
    allsum=0.0;
    allcot = 0;
    for i in range(ust_shape[0]):
        for j in range(ust_shape[1]):
            ussum=0;uscot=0;
            for k in range(ust_shape[2]):
                tmp = ust_mat[i,j,k];
                if tmp != NoneValue:
                    uscot+=1;ussum+=tmp;
                else:
                    nv_locs.append((i,j,k));
            allsum+=ussum;
            allcot+=uscot;
            if uscot!= 0:
                us_mean[i,j] = ussum / uscot*1.0;
    
    us_mean_path = dataset_out_path + r'/us_mean.txt';        
    np.savetxt(us_mean_path, us_mean, '%.2f');        
    print ('统计计算接收，耗时 %.2f秒  ,有效总和为%.2f,有效计数为%d,无效计数为%d\n'%((time.time() - tnow),allsum,allcot,len(nv_locs)));

    print ('0填充并分割开始');
    tnow = time.time();
    for fi in nv_locs:
        ust_mat[fi]=randomin(0.0);
    split(types_name[0],0);
    print ('0填充并分割结束，耗时 %.2f秒  \n'%((time.time() - tnow))); 

    print ('全局有效平均填充并分割开始');
    tnow = time.time();
    am = allsum /allcot;
    for fi in nv_locs:
        ust_mat[fi]=randomin(am);
    split(types_name[1],1);
    print ('全局有效平均填充并分割结束，耗时 %.2f秒  ,有效平均值为 %.2f\n'%((time.time() - tnow),am)); 

    print ('us有效平均填充并分割开始');
    tnow = time.time();
    for fi in nv_locs:
        tmp = us_mean[fi[0],fi[1]];
        if tmp != NoneValue: 
            ust_mat[fi]=randomin(tmp);
        else:
            ust_mat[fi]=randomin(am);        
    split(types_name[2],2);
    print ('us有效平均填充并分割开始，耗时 %.2f秒  \n'%((time.time() - tnow))); 

    print ('任务完成，总耗时%d秒 '%(time.time() - now));

    
#    print(ust_mat);
#     print(ust_mat.shape);
#     print(nv_locs);

if __name__ == '__main__':
    spilter();
    pass