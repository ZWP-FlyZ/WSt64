# -*- coding: utf-8 -*-
'''
Created on 2018年4月27日

@author: zwp
'''

'''
    数据预处理
'''

import numpy as np;
from mf.MFS import MF_bl as mf;
import random;
def removeNoneValue(R):
    '''
    清除无效的数据
    '''
    if R is None:
        return R;
    ind = np.where(R<0);
    R[ind]=0;



def preprocess(R):
    pass;

def preprocess1(R,isUAE = True):
    '''
    输入原始数据集
    将无效数据替换成0
    '''
    
    ind = np.where(R>0);
    newR = np.zeros_like(R);
    newR[ind]=1;
    if isUAE:
        sum_arr = np.sum(newR,axis=0);
    else:
        sum_arr = np.sum(newR,axis=1);
        R = R.T;
        
    batch_size = R.shape[0];    
    most = int(np.median(sum_arr));
    feat_ind = np.where(sum_arr==0)[0];
    for fid in feat_ind:
        batch_ind = np.random.randint(0,batch_size,size=most);
        choosed = R[batch_ind];
        tmp = np.count_nonzero(choosed, axis=1);
        means = np.sum(choosed,axis=1);
        means /=tmp;
        R[batch_ind,[fid]*most]=means;

    if not isUAE:
        R = R.T;
    
def preprocess2(R,isUAE = True,mr_mut=2):
    '''
    输入原始数据集
    将无效数据替换成0
    '''
    
    ind = np.where(R>0);
    newR = np.zeros_like(R);
    newR[ind]=1;
    if isUAE:
        sum_arr = np.sum(newR,axis=0);
    else:
        sum_arr = np.sum(newR,axis=1);
        R = R.T;
        
    batch_size = R.shape[0];    
    most = int(np.median(sum_arr));
    mean = np.mean(sum_arr);
    std = np.std(sum_arr);
    delta = int(mean - mr_mut*std)+1;
    if delta<1:delta = 1; 
    
    for edge in range(delta):
        
        feat_ind = np.where(sum_arr==edge)[0];
        for fid in feat_ind:
            cmp_size = most-edge;
            batch_ind = np.random.randint(0,batch_size,size=cmp_size);
            choosed = R[batch_ind];
            tmp = np.count_nonzero(choosed, axis=1);
            means = np.sum(choosed,axis=1);
            means /=tmp;
            R[batch_ind,[fid]*cmp_size]=means;

    if not isUAE:
        R = R.T;

def preprocessMF(R,mf,isUAE = True,mr_mut=2):
    '''
    由矩阵分解提供填补值
    '''
    ind = np.where(R>0);
    newR = np.zeros_like(R);
    newR[ind]=1;
    if isUAE:
        sum_arr = np.sum(newR,axis=0);
    else:
        sum_arr = np.sum(newR,axis=1);
        R = R.T;
        
    batch_size = R.shape[0];    
    most = int(np.median(sum_arr));
    mean = np.mean(sum_arr);
    std = np.std(sum_arr);
    delta = int(mean - mr_mut*std)+1;
    if delta<1:delta = 1; 
    
    for edge in range(delta):
        feat_ind = np.where(sum_arr==edge)[0];
        for fid in feat_ind:
            cmp_size = most-edge;
            batch_ind = np.random.randint(0,batch_size,size=cmp_size);
            for bid in batch_ind:
                if isUAE:
                    R[bid,fid]=mf.predict(bid, fid);
                else:
                    R[bid,fid]=mf.predict(fid,bid);
    if not isUAE:
        R = R.T;    
    pass;

def preprocessMF_rat(R,mf,isUAE = True,rat=0.0):
    '''
    由矩阵分解提供填补值
    '''
    if not isUAE:
        R = R.T;    
    batch_size = R.shape[0];
    feat_size = R.shape[1];
    sum_arr = np.zeros((feat_size,),int);
    contain_sets=[[] for _ in range(feat_size)];
    none_ind = np.argwhere(R>0);
    for bid,fid in none_ind:
        contain_sets[fid].append(bid);
        sum_arr[fid]+=1;
        
    most = int(np.median(sum_arr));
    if rat<=0.0:
        delta = 1;
        top = most;
    else:
        top = int(rat*batch_size);
        delta = top;
    
    all_range_set=set(range(batch_size));
    for edge in range(delta):
        feat_ind = np.where(sum_arr==edge)[0];
        for fid in feat_ind:
            rem_set = all_range_set-set(contain_sets[fid]);
            cmp_size = top-edge;
            batch_ind = random.sample(rem_set,cmp_size);
            for bid in batch_ind:
                if isUAE:
                    R[bid,fid]=mf.predict(bid, fid);
                else:
                    R[bid,fid]=mf.predict(fid,bid);
    if not isUAE:
        R = R.T;    
    pass;




if __name__ == '__main__':
    pass