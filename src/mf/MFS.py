# -*- coding: utf-8 -*-
'''
Created on 2018年4月28日

@author: zwp
'''

import numpy as np;
import time;
import math;
import os;

rou = 0.05;

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
            'P':np.random.normal(0,rou,(us_shape[0],size_f))/np.sqrt(size_f),
            'Q':np.random.normal(0,rou,(us_shape[1],size_f))/np.sqrt(size_f),
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

    def value_optimize(self,u,i,rt,lr,lamda):
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
        
    def train_mat(self,R,repeat,learn_rate,lamda=0.02,save_path=None):
        print('|-->训练开始，learn_rate=%f,repeat=%d'%(learn_rate,repeat));
        now = time.time();
        shp = R.shape;
        cal_set=[];
        for u in range(shp[0]):
            for i in range(shp[1]):        
                if R[u,i]!=0:
                    cal_set.append((u,i));
        cot=len(cal_set);
        for rep in range(repeat):
            tnow=time.time();
            maeAll=0.0;rmseAll=0.0;
            for ui in cal_set:
                rt = R[ui];
                pt = self.value_optimize(ui[0], ui[1], rt, learn_rate,lamda);
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

class MF_bl_plus:
    us_shape=None;
    size_f = None;
    mean = None;
    values = None;
    Iu_num = None;
    def __init__(self,us_shape,size_f,mean,Iu_num):
        self.us_shape = us_shape;
        self.size_f = size_f;
        self.mean = mean;
        self.Iu_num = Iu_num;
        self.values={
            'P':np.random.normal(0,rou,(us_shape[0],size_f))/np.sqrt(size_f),
            'Q':np.random.normal(0,rou,(us_shape[1],size_f))/np.sqrt(size_f),
            'bu':np.zeros(us_shape[0]),
            'bi':np.zeros(us_shape[1]),
            'yi':[np.zeros((num,)) for num in Iu_num]           
        };
        
    def predict(self,u,i):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        yi = self.values['yi'];
        res=self.mean+bu[u]+bi[i];
        pu = P[u] + np.sum()
        res += np.sum(P[u]*Q[i]);
        return res;

    def value_optimize(self,u,i,rt,lr,lamda):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        yi = self.values['yi'];
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
        self.values['yi']=yi;
        return pt;
        
    def train_mat(self,R,repeat,learn_rate,lamda=0.02,save_path=None):
        print('|-->训练开始，learn_rate=%f,repeat=%d'%(learn_rate,repeat));
        now = time.time();
        shp = R.shape;
        cal_set=[];
        for u in range(shp[0]):
            for i in range(shp[1]):        
                if R[u,i]!=0:
                    cal_set.append((u,i));
        cot=len(cal_set);
        for rep in range(repeat):
            tnow=time.time();
            maeAll=0.0;rmseAll=0.0;
            for ui in cal_set:
                rt = R[ui];
                pt = self.value_optimize(ui[0], ui[1], rt, learn_rate,lamda);
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
        if os.path.exists(path+'/yi.txt'):
            self.values['yi']=np.loadtxt(path+'/yi.txt', float);
                       
    def saveValues(self,path):
        np.savetxt(path+'/P.txt',self.values['P'],'%.6f');
        np.savetxt(path+'/Q.txt',self.values['Q'],'%.6f');
        np.savetxt(path+'/bu.txt',self.values['bu'],'%.6f');
        np.savetxt(path+'/bi.txt',self.values['bi'],'%.6f');
        np.savetxt(path+'/yi.txt',self.values['yi'],'%.6f');
          
    def exisValues(self,path,isUAE=True):
        if not os.path.exists(path+'/Q.txt'):
            return False
        if not os.path.exists(path+'/P.txt'):
            return False        
        if not os.path.exists(path+'/bu.txt'):
            return False        
        if not os.path.exists(path+'/bi.txt'):
            return False
        if not os.path.exists(path+'/yi.txt'):
            return False        
        return True;
    def getValues(self):
        return self.values;    
   
############################ end class ###########################

