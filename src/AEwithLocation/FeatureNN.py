# -*- coding: utf-8 -*-
'''
Created on 2018年4月11日

@author: zwp
'''




import numpy as np;
import os ;


class HidNN():
    '''
    负责获取隐含特征的扩展数据
    '''
    
    base_path = '';# 缓存路径
    
    isUAE = True; # 是否基于用户
    
    R = None;# 原始数据集
    
    R_p = None;# 数据集参数
    
    tag_shape = None;
    # 根据isUAE变化选择[batch,s] 或者 [batch,u]
    
    W = None;
    # 相似度矩阵 shape = [batch,batch]
    # W的aixe=0表示origin，aixe=1表示target 
    
    C = None;
    # 互补率矩阵 shape = [batch,batch]
    
    
    
    
    def __init__(self,
                path,# 缓存路径
                isUAE,# 基于用户的自编码方法
                R,# 数据集shape=[u,s]
                R_p# 数据集参数[spa,case,NoneValue]
                ):
        if not os.path.isdir(path):
            os.mkdir(path);
        self.base_path=path;
        self.isUAE = isUAE;
        self.R = R;
        self.R_p = R_p;
        if isUAE:
            self.tag_shape = R.shape;
        else:
            sp = R.shape;
            self.tag_shape =[sp[1],sp[0]];
        
        self.init_param(); 
        pass;
    
    def init_param(self):
        '''
        初始化
        '''
        wc = [self.get_cache_name('W'),
              self.get_cache_name('C')];
        if self.checkCache(wc[0]) and \
            self.checkCache(wc[1]):
            self.W = np.loadtxt(self.toCachePath(wc[0]),float);
            self.C = np.loadtxt(self.toCachePath(wc[1]),float);
        else:
            self.calculteWC();
        pass;
    
    
    
    def calculteWC(self):
        sp0 = self.tag_shape[0];
        NoneValue = self.R_p[2];
        W = np.zeros([sp0,sp0]);
        C = np.zeros_like(W);
        R = self.R;
        if not self.isUAE:
            R = R.T;
        print('开始计算W与C');
        for i in range(sp0):
            a = R[i];
            b = R;
            # 计算距离

            alog = a!=NoneValue;
            blog = b!=NoneValue;
            delta=np.subtract(b,a,out=np.zeros_like(R),where=(alog & blog));

#             delta=np.subtract(b,a,out=np.zeros_like(R),where=(alog));
            
            W[i]=np.sqrt(np.sum(delta**2,axis=1));
            
            # 计算互补量
            tag_n_size = np.alen(np.where(a!=NoneValue)[0]);
            div = np.divide(b,a,out=np.zeros_like(R),where=a!=NoneValue);
            gt = np.where(div>0);
            div[gt]=1.0;
            div = np.sum(div,axis=1);
            C[:,i]=(tag_n_size-div)*1.0/tag_n_size;
            if i % 50 == 0:
                print('--->ws_step%d'%(i));
        self.W=W;
        self.C=C;
        w_path = self.toCachePath(self.get_cache_name('W'));
        c_path = self.toCachePath(self.get_cache_name('C'));
        np.savetxt(w_path,W,'%.20f');
        np.savetxt(c_path,C,'%.6f');
        pass;
    
    def getExtendDataIndex(self,ori_c_index,tag_c_index,k):
        '''
        获取拓展数据
        ori_c_index 原簇index列表
        tag_c_index 目标簇index列表
        ori_c_index与tag_c_index需要互斥
        k 获取的扩展数量
        返回扩展数据的index列表shape=[k]
        '''
        ori_cluster_w = self.W[ori_c_index,:];
        ori_cluster_c = self.C[ori_c_index,:];
        ot_cluster_w = ori_cluster_w[:,tag_c_index];
        ot_cluster_c = ori_cluster_c[:,tag_c_index];
        
        avg_ot_w = np.average(ot_cluster_w,axis=0);
        
        args = np.argsort(avg_ot_w)[0:k];
        return tag_c_index[args].tolist();

    def getExtendDataIndex2(self,ori_c_index,tag_c_index,k):
        '''
        获取拓展数据
        ori_c_index 原簇index列表
        tag_c_index 目标簇index列表
        ori_c_index与tag_c_index需要互斥
        k 获取的扩展数量
        返回扩展数据的index列表shape=[k]
        '''
        ori_cluster_w = self.W[ori_c_index,:];
        ori_cluster_c = self.C[ori_c_index,:];
        ot_cluster_w = ori_cluster_w[:,tag_c_index];
        ot_cluster_c = ori_cluster_c[:,tag_c_index];
        
        sorted_args = np.argsort(ot_cluster_w,axis=1)[:,0:k];
        sorted_args=sorted_args.reshape((-1,));
        p_tmp = np.zeros(len(tag_c_index));
        for i in sorted_args:
            p_tmp[i]=p_tmp[i]+1;
        args = np.argsort(-p_tmp);
        return tag_c_index[args[0:k]].tolist();    
    
    def getExtendDataIndex3(self,ori_c_index,tag_c_index,k):
        '''
        获取拓展数据
        ori_c_index 原簇index列表
        tag_c_index 目标簇index列表
        ori_c_index与tag_c_index需要互斥
        k 获取的扩展数量
        返回扩展数据的index列表shape=[k]
        '''
        ori_cluster_w = self.W[ori_c_index,:];
        ori_cluster_c = self.C[ori_c_index,:];
        ot_cluster_w = ori_cluster_w[:,tag_c_index];
        ot_cluster_c = ori_cluster_c[:,tag_c_index];
        
        sorted_args = np.argsort(ot_cluster_w,axis=1)[:,0:k];
        sorted_args=sorted_args.reshape((-1,));
        p_tmp = np.zeros(len(tag_c_index));
        cot=0;
        for i in sorted_args:
            p_tmp[i]=p_tmp[i]+1.0/(2+cot%k);
            cot+=1;
        args = np.argsort(-p_tmp);
        return tag_c_index[args[0:k]].tolist();    
    
    
    def get_cache_name(self,p_name):
        '''
        返回一个缓存文件名：
        (p_name)_isUAE_(isUAE)_spa(R_p[0])_case(R_p[1]).txt
        '''
        return '%s_isUAE_%s_spa%d_case%d.txt'%(p_name,self.isUAE,\
                                               self.R_p[0],self.R_p[1]);
    
    def toCachePath(self,cache_fn):
        return self.base_path+'/'+cache_fn;
    def checkCache(self,cache_fn):
        '''
        检查cache_fn文件是否存在，
        如果存在加载缓存，若不存在结束并返回False
        加载结束返回True
        '''
        return os.path.exists(self.toCachePath(cache_fn));
    
    
        


