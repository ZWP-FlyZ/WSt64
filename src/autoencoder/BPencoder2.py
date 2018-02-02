# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日

@author: zwp12
'''

'''

BPencoder 代码思路来自论文《基于自编码器的评分预测算法》

在us矩阵中，空值用NoneVale 替代，在神经网络训练过程中，
跳过对NoneValue对应节点参数的调整

'''

import numpy as np;
import time;
import math;
import os;




class BPAutoEncoder:
    
    def __init__(self,X_n,hidden_n,active_function,de_actfunc,check_none):
        self.size_x = X_n;
        self.size_hidden=hidden_n;
        self.func = active_function;
        self.defunc=de_actfunc;
        self.check_none= check_none;
        self.values= {
            'w1':np.random.normal(0,rou,(self.size_x,self.size_hidden)),
            'w2':np.random.normal(0,rou,(self.size_hidden,self.size_x)),
            'b1':np.random.normal(0,rou,(1,self.size_hidden)),
            'b2':np.random.normal(0,rou,(1,self.size_x)),
            'h':None
            };
    
    ## 注意这里计算过方法必须保证NoneValue=0
    def calculate(self,x):
        x = np.reshape(x, (1,self.size_x));
        h = self.func(np.matmul(x,self.values['w1'])+self.values['b1']);
        self.values['h']=h.reshape(self.size_hidden);
        y = self.func(np.matmul(h,self.values['w2'])+self.values['b2']);
        return y.reshape(self.size_x,);

    def calFill(self,R):
        for j in range(R.shape[1]):
            py = self.calculate(R[:,j]);
            R[:,j] = py;
        return R;

    
    def evel(self,py,y):
        mae=0.0;rmse=0.0;
        cot=0;
        for i in range(self.size_x):
            if self.check_none(y[i]):
                continue;
            cot+=1;
            delta=abs(y[i]-py[i]);
            mae+=delta;
            rmse+=delta**2;
        if cot==0:
            return (-1,-1);
        else:
            return (mae/cot,math.sqrt(rmse/cot));
    
    # 更新参数
    def layer_optimize(self,py,y):
        b1 = self.values['b1'];
        w1 = self.values['w1'];
        b2 = self.values['b2'];
        w2 = self.values['w2'];
        h  = self.values['h'];
        lr = self.lr;
        gjs=np.zeros(self.size_x);
        for j in range(self.size_x):
            if self.check_none(y[j]):
                continue;
            k = py[j];
            gj=(k-y[j])*self.defunc(k);
            gjs[j]=gj;
            k = lr*gj;
            b2[0,j]=b2[0,j]-k;
            w2[:,j]=w2[:,j]-k*h;

        gis = np.zeros(self.size_hidden);
        for i in range(self.size_hidden):
            k=np.sum(gjs*w2[i,:]);
            k=self.defunc(h[i])*k;
            b1[0,i]=b1[0,i]-lr*k;
            gis[i]=k;
            
        tmp =lr*gis;
        for k in range(self.size_x):
            if self.check_none(y[k]):
                continue;
            w1[k,:]=w1[k,:]-tmp*y[k];            
            
        self.values['b2']=b2;
        self.values['w2']=w2;        
        self.values['b1']=b1;
        self.values['w1']=w1;
                        

    def train(self,X,learn_rate,repeat):
        self.lr=learn_rate;
        X = X.T;
        print('-->训练开始，learn_rate=%f,repeat=%d \n'%(learn_rate,repeat));
        now = time.time();
        for rep in range(repeat):
            tnow=time.time();
            # self.lr=self.lr*0.9;
            maeAll=0.0;rmseAll=0.0;
            shape1=X.shape[0];
            for i in range(shape1):
                x = X[i];
                py = self.calculate(x);
                self.layer_optimize(py,x);
                mae,rmse=self.evel(py, x);
                maeAll+=mae/shape1;
                rmseAll+=rmse/shape1;
            print('---->step%d 耗时%.2f秒 MAE=%.3f RMSE=%.3f'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
        print('\n-->训练结束，总耗时%.2f秒  learn_rate=%.3f,repeat=%d \n'%((time.time()-now),learn_rate,repeat));
        
        
    def preloadValues(self,path):
        if os.path.exists(path+'/w1.txt'):
            self.values['w1']=np.loadtxt(path+'/w1.txt', float);
        if os.path.exists(path+'/w2.txt'):
            self.values['w2']=np.loadtxt(path+'/w2.txt', float);        
        if os.path.exists(path+'/b1.txt'):
            self.values['b1']=np.loadtxt(path+'/b1.txt', float);        
        if os.path.exists(path+'/b2.txt'):
            self.values['b2']=np.loadtxt(path+'/b2.txt', float);
        if os.path.exists(path+'/h.txt'):
            self.values['h']=np.loadtxt(path+'/h.txt', float);            
    def saveValues(self,path):
        np.savetxt(path+'/w1.txt',self.values['w1'],'%.6f');
        np.savetxt(path+'/w2.txt',self.values['w2'],'%.6f');
        np.savetxt(path+'/b1.txt',self.values['b1'],'%.6f');
        np.savetxt(path+'/b2.txt',self.values['b2'],'%.6f');
        np.savetxt(path+'/h.txt',self.values['h'],'%.6f');    
    
    def getValues(self):
        return self.values;    
############################ end class ###########################    
    



def active_function(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)));


def de_active_function(x):
    return x*(1.0-x);

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



base_path = r'E:/Dataset/wst';
origin_data = base_path+'/rtdata.txt';

us_shape=(339,5825);

mean = 0.908570086101;
ek = 1.9325920405;

hidden_node = 100;
learn_rate=0.05;
repeat = 12;
rou=0.2

R=None;
    
case = 3;
NoneValue = 0.0;


loadvalues= False;

def encoder_run(spa):
    train_data = r'E:/Dataset/ws/train/sparseness%d/training%d.txt'%(spa,case);
    test_data = r'E:/Dataset/ws/test/sparseness%d/test%d.txt'%(spa,case);
    
       
    values_path=r'E:/Dataset/ws';
    
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
    R=preprocess(R);
    print(R);
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    encoder = BPAutoEncoder(us_shape[0],hidden_node,
                            active_function,de_active_function,check_none);
    if loadvalues :
        encoder.preloadValues(values_path);
    else:
        encoder.train(R, learn_rate, repeat);
    encoder.saveValues(values_path);
    print ('训练模型开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  
    
        
    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    PR = encoder.calFill(R);
    mae=0.0;rmse=0.0;cot=0;

    for tc in trdata:
        if tc[2]<=0.0:
            continue;
        delta = PR[int(tc[0]),int(tc[1]-1)]*20-tc[2];    
        mae+=abs(delta);
        rmse+=(delta)**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.3f,RMSE=%.3f\n'%((time.time()-now),spa,mae,rmse));

        
if __name__ == '__main__':
    encoder_run(5);
    pass