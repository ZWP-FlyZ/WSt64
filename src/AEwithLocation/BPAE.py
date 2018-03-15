# -*- coding: utf-8 -*-
'''
Created on 2018年3月15日

@author: zwp
'''
import numpy as np;
import math;
import time;
import os;

rou = 0.1;

class BPAutoEncoder:
    
    def __init__(self,X_n,hidden_n,actfun1,deactfun1,actfun2,deactfun2,check_none):
        self.size_x = X_n;
        self.size_hidden=hidden_n;
        self.func1 = actfun1;
        self.defunc1 =deactfun1;
        self.func2 = actfun2;
        self.defunc2 =deactfun2;
        self.check_none= check_none;
        self.values= {
            'w1':np.random.normal(0,rou,(self.size_x,self.size_hidden))/np.sqrt(hidden_n),
            'w2':np.random.normal(0,rou,(self.size_hidden,self.size_x))/np.sqrt(hidden_n),
            'b1':np.random.normal(0,rou,(1,self.size_hidden))/np.sqrt(hidden_n),
            'b2':np.random.normal(0,rou,(1,self.size_x))/np.sqrt(hidden_n),
            'h':None
            };
    
    ## 注意这里计算过方法必须保证NoneValue=0
    def calculate(self,x):
        x = np.reshape(x, (1,self.size_x));
        h = self.func1(np.matmul(x,self.values['w1'])+self.values['b1']);
        self.values['h']=h.reshape(self.size_hidden);
        y = self.func2(np.matmul(h,self.values['w2'])+self.values['b2']);
        return y.reshape(self.size_x,);

    def calFill(self,R):
        PR = np.zeros(R.shape,float);
        for j in range(R.shape[1]):
            py = self.calculate(R[:,j]);
            PR[:,j] = py;
        return PR;

    
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
            gj=(k-y[j])*self.defunc2(k);
            gjs[j]=gj;
            k = lr*gj;
            b2[0,j]=b2[0,j]-k;
            w2[:,j]=w2[:,j]-k*h;

        gis = np.zeros(self.size_hidden);
        for i in range(self.size_hidden):
            k=np.sum(gjs*w2[i,:]);
            k=self.defunc1(h[i])*k;
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
                        

    def train(self,X,learn_rate,repeat,save_path=None):
        self.lr=learn_rate;
        X = X.T;
        print('-->训练开始，learn_rate=%f,repeat=%d \n'%(learn_rate,repeat));
        now = time.time();
        for rep in range(repeat):
            tnow=time.time();
            #self.lr=self.lr*0.95;
            maeAll=0.0;rmseAll=0.0;
            shape1=X.shape[0];
            for i in range(shape1):
                x = X[i];
                py = self.calculate(x);
                mae,rmse=self.evel(py, x);
                self.layer_optimize(py,x);
                maeAll+=mae/shape1;
                rmseAll+=rmse/shape1;
#             print(py);
            if save_path != None:
                self.saveValues(save_path);
            print('---->step%d 耗时%.2f秒 MAE=%.6f RMSE=%.6f'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
        print('\n-->训练结束，总耗时%.2f秒  learn_rate=%.3f,repeat=%d \n'%((time.time()-now),learn_rate,repeat));
 
        
    def preloadValues(self,path,isUAE=True):
        if os.path.exists(path+'/w1_%s.txt'%(isUAE)):
            self.values['w1']=np.loadtxt(path+'/w1_%s.txt'%(isUAE), np.float64);
        if os.path.exists(path+'/w2_%s.txt'%(isUAE)):
            self.values['w2']=np.loadtxt(path+'/w2_%s.txt'%(isUAE), np.float64);        
        if os.path.exists(path+'/b1_%s.txt'%(isUAE)):
            self.values['b1']=np.loadtxt(path+'/b1_%s.txt'%(isUAE), np.float64).reshape(1,self.size_hidden);        
        if os.path.exists(path+'/b2_%s.txt'%(isUAE)):
            self.values['b2']=np.loadtxt(path+'/b2_%s.txt'%(isUAE), np.float64).reshape(1,self.size_x);
        if os.path.exists(path+'/h_%s.txt'%(isUAE)):
            self.values['h']=np.loadtxt(path+'/h_%s.txt'%(isUAE), np.float64);            
    def saveValues(self,path,isUAE=True):
        np.savetxt(path+'/w1_%s.txt'%(isUAE),self.values['w1'],'%.12f');
        np.savetxt(path+'/w2_%s.txt'%(isUAE),self.values['w2'],'%.12f');
        np.savetxt(path+'/b1_%s.txt'%(isUAE),self.values['b1'],'%.12f');
        np.savetxt(path+'/b2_%s.txt'%(isUAE),self.values['b2'],'%.12f');
        np.savetxt(path+'/h_%s.txt'%(isUAE),self.values['h'],'%.12f');    
    def exisValues(self,path,isUAE=True):
        if not os.path.exists(path+'/w1_%s.txt'%(isUAE)):
            return False;
        if not os.path.exists(path+'/w2_%s.txt'%(isUAE)):
            return False;        
        if not os.path.exists(path+'/b1_%s.txt'%(isUAE)):
            return False;      
        if not os.path.exists(path+'/b2_%s.txt'%(isUAE)):
            return False;
        if not os.path.exists(path+'/h_%s.txt'%(isUAE)):
            return False;        
        return True;
    def getValues(self):
        return self.values;    
   
############################ end class ###########################



