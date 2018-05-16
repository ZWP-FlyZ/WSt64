# -*- coding: utf-8 -*-
'''
Created on 2018年5月14日

@author: zwp
'''

import numpy as np;
from mf import MFS;
import os;
class mf_adaboost():
    '''
    使用矩阵分解adaboost方法
    '''
    R = None;# 原始数据集,US矩阵
    K = 0;# Adaboost 迭代次数
    end_err = 0.0;# 结束迭代的误差数
    
    DW = None;# 权重矩阵
    AdaModels = [];# 模型列表
    Ak=None;# 权重列表
    
    batch_size = 0;# 数据总数目
    R_list = None;
    
    predict_param = None;# 
    newK = 0;
    def __init__(self,R,K,end_err=0.0):
        '''
        R 元数据集
        K 迭代次数
        end_err 结束误差
        '''
        self.R = R;
        self.K = K;
        self.end_err = end_err;
        ind = np.argwhere(R>0);
        self.batch_size = len(ind);
        self.DW = np.ones((self.batch_size,),float)/self.batch_size;
        self.R_list=[[u,s,R[u,s]] for u,s in ind];
        pass;
    
    def train(self,size_f,repeat,learn_rate,lamda,kset=None):
        '''
        size_f 隐特征数
        '''
        AdaModels = [];
        Ak = [];
        
        R = self.R;
        DW = self.DW;
        if kset==None:kset=[0,self.K];
        mean = np.sum(R)/self.batch_size;
        for k in range(*kset):
            model_k = MFS.MF_bl_adaboost(R.shape,size_f,mean);
            model_k.train_mat(self.R_list,repeat[k],learn_rate[k],lamda[k],DW);
            
            err_list = [];
            for exm in self.R_list:
                pt = model_k.predict(exm[0], exm[1]);
                err_list.append(abs(pt-exm[2]));
            err_list = np.array(err_list);
            mae = np.mean(err_list);
            Ek = np.max(err_list);
            # 采用平方误差法
            err_list = (err_list/Ek)**1;
            # 该次迭代误差
            ek = np.sum(DW*err_list);
            # k的权重
            ak = ek / (1-ek);
            # 更新DW
            DW = DW * np.power(ak,1-err_list);
            DW = DW / np.sum(DW);
            Ak.append(ak);
            AdaModels.append(model_k);
            print('step%d ek=%f ak=%f mae=%f'%(k+1,ek,ak,mae));
            if mae <= self.end_err:break;
        
        self.DW = DW;
        self.Ak = np.array(Ak);
        self.AdaModels = AdaModels;     
        self.predict_param=np.sum(np.log(1.0/self.Ak));
        
    
    def update_train(self,size_f,repeat,learn_rate,lamda):
        
        '''
        若newK>K ,则继续增量训练
        若newK<=K,则不变
        '''
        
        if self.newK> self.K:
            self.train(size_f, repeat, learn_rate, lamda, [self.K,self.newK]);
            self.K = self.newK;
        else:
            self.newK = self.K;
    
    
        
    def predict(self,u,i):
        Ak = self.Ak;
        AdaModels = self.AdaModels;
        preparam = self.predict_param;
        tmp = [model.predict(u,i) for model in AdaModels];
        tmp = np.array(tmp)*Ak;
        gx = np.median(tmp);
        return gx * preparam;
#         AK = 1.0 / Ak;
#         tmp = np.array(tmp)*Ak;        
#         return np.sum(tmp)/np.sum(Ak);
    
    def save_param(self,path):
        if not os.path.isdir(path):
            os.mkdir(path);
        import pickle;
        fw = open(path+'/mfadaboost','bw');
        obj = (self.K,self.DW,self.AdaModels,self.Ak);
        pickle.dump(obj,fw);
        fw.close();
            
    def load_param(self,path):
        import pickle;
        fr = open(path+'/mfadaboost','br');
        K,DW,AdaMod,Ak = pickle.load(fr); 
        self.newK = self.K;
        self.K = K;
        self.DW=DW;
        self.AdaModels = AdaMod;
        self.Ak=Ak;
        self.predict_param=np.sum(np.log(1.0/self.Ak));        
    
    pass;







