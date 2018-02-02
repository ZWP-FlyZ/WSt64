# -*- coding: utf-8 -*-
'''
Created on 2018年2月1日

@author: zwp12
'''

'''
将矩阵分解和基准线预测结合

'''

import numpy as np;
import time;
import math;
import os;


class MF_bl:
    us_shape=None;
    size_f = None;
    mean = None;
    values = None;
    def __init__(self,us_shape,size_f,mean):
        self.us_shape = us_shape;
        self.size_f = size_f;
        self.mean = mean;
        self.values={
            'P':np.random.normal(0,rou,(us_shape[0],size_f)),
            'Q':np.random.normal(0,rou,(us_shape[1],size_f)),
            'bu':np.zeros(us_shape[0]),
            'bi':np.zeros(us_shape[1])           
        };
        
    def predict(self,u,i):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        res=self.mean+bu[u]+bi[i];
        res += np.sum(P[u]*Q[i]);
        return res;

    def value_optimize(self,u,i,rt,lr):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        pt =  self.predict(u, i);       
        eui = rt-pt;
        # 更改baseline 偏置项
        bu[u] += lr * (eui-lamda*bu[u]); 
        bi[i] += lr * (eui-lamda*bi[i]);
        #更改MF 参数
        tmp = lr * (eui*Q[i]-lamda*P[u]);
        Q[i] += lr * (eui*P[u]-lamda*Q[i]);
        P[u]+=tmp;
        self.values['P']=P;
        self.values['Q']=Q;
        self.values['bu']=bu;
        self.values['bi']=bi;
        return pt;
        
    def train_mat(self,R,repeat,learn_rate,save_path=None):
        print('|-->训练开始，learn_rate=%f,repeat=%d'%(learn_rate,repeat));
        now = time.time();
        shp = R.shape;
        cal_set=[];
        for u in range(shp[0]):
            for i in range(shp[1]):        
                if R[u,i]!=NoneValue:
                    cal_set.append((u,i));
        cot=len(cal_set);
        for rep in range(repeat):
            tnow=time.time();
            maeAll=0.0;rmseAll=0.0;
            for ui in cal_set:
                rt = R[ui];
                pt = self.value_optimize(ui[0], ui[1], rt, learn_rate);
                maeAll+=abs(rt-pt);
            maeAll = maeAll / cot;         
            if save_path != None:
                self.saveValues(save_path);
            print('|---->step%d 耗时%.2f秒 MAE=%.6f RMSE=%.6f|'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
        print('|-->训练结束，总耗时%.2f秒  learn_rate=%.3f,repeat=%d \n'%((time.time()-now),learn_rate,repeat));


    def preloadValues(self,path):
        if os.path.exists(path+'/Q.txt'):
            self.values['Q']=np.loadtxt(path+'/Q.txt', float);
        if os.path.exists(path+'/P.txt'):
            self.values['P']=np.loadtxt(path+'/P.txt', float);        
        if os.path.exists(path+'/bu.txt'):
            self.values['bu']=np.loadtxt(path+'/bu.txt', float);        
        if os.path.exists(path+'/bi.txt'):
            self.values['bi']=np.loadtxt(path+'/bi.txt', float);
           
    def saveValues(self,path):
        np.savetxt(path+'/P.txt',self.values['P'],'%.6f');
        np.savetxt(path+'/Q.txt',self.values['Q'],'%.6f');
        np.savetxt(path+'/bu.txt',self.values['bu'],'%.6f');
        np.savetxt(path+'/bi.txt',self.values['bi'],'%.6f');
  
    def exisValues(self,path,isUAE=True):
        if not os.path.exists(path+'/Q.txt'):
            return False
        if not os.path.exists(path+'/P.txt'):
            return False        
        if not os.path.exists(path+'/bu.txt'):
            return False        
        if not os.path.exists(path+'/bi.txt'):
            return False       
        return True;
    def getValues(self):
        return self.values;    
   
############################ end class ###########################    

def preprocess(R):
    if R is None:
        return R;
    shape = R.shape;
    cot=0;
    sum=0.0;
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp = R[i,j];
            if tmp<0.0:
                R[i,j] = 0.0;
            elif tmp>0.0:
                sum+=tmp; 
                cot+=1;
    #return  (R -  mean) / ek;
    return  R, sum/cot;




NoneValue= 0;
# 初始化参数中 正态分布标准差
rou = 0.1;
# 在矩阵分解中 正则化 参数
lamda = 0.005;

# 隐属性数
f = 20;

#训练次数
repeat = 20;

# 学习速率
learn_rate = 0.01;


us_shape=(339,5825);
case = 1;
loadvalues=False;
continue_train=True;

def mf_base_run(spa,case):
    train_data = r'E:/Dataset/ws/train/sparseness%d/training%d.txt'%(spa,case);
    test_data = r'E:/Dataset/ws/test/sparseness%d/test%d.txt'%(spa,case);
       
    values_path=r'E:/Dataset/mf_baseline_values/spa%d'%(spa);
    
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
    R,mean=preprocess(R);
    print(R,mean);
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    tx = us_shape[0];

    svd = MF_bl(R.shape,f,mean);

    if loadvalues and svd.exisValues(values_path):
        svd.preloadValues(values_path);
    if continue_train:
        svd.train_mat(R, repeat,learn_rate,values_path);
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
    mf_base_run(20,case)
    
    pass