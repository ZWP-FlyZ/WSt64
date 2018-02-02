# -*- coding: utf-8 -*-
'''
Created on 2018年1月26日

@author: zwp12
'''

import numpy as np;
import time ;
import os;


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
            'w1':np.random.normal(0,rou,(self.size_x,self.size_hidden)),
            'w2':np.random.normal(0,rou,(self.size_hidden,self.size_x)),
            'b1':np.random.normal(0,rou,(1,self.size_hidden)),
            'b2':np.random.normal(0,rou,(1,self.size_x)),
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
            return (mae/cot,np.sqrt(rmse/cot));
    
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
                        

    def train(self,X,learn_rate,repeat):
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
            # self.saveValues('E:/Dataset/ws');
            print('---->step%d 耗时%.2f秒 MAE=%.3f RMSE=%.3f'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
        print('\n-->训练结束，总耗时%.2f秒  learn_rate=%.3f,repeat=%d \n'%((time.time()-now),learn_rate,repeat));
        
        
    def preloadValues(self,path):
        if os.path.exists(path+'/w1_%s.txt'%(isUserAutoEncoder)):
            self.values['w1']=np.loadtxt(path+'/w1_%s.txt'%(isUserAutoEncoder), float);
        if os.path.exists(path+'/w2_%s.txt'%(isUserAutoEncoder)):
            self.values['w2']=np.loadtxt(path+'/w2_%s.txt'%(isUserAutoEncoder), float);        
        if os.path.exists(path+'/b1_%s.txt'%(isUserAutoEncoder)):
            self.values['b1']=np.loadtxt(path+'/b1_%s.txt'%(isUserAutoEncoder), float);        
        if os.path.exists(path+'/b2_%s.txt'%(isUserAutoEncoder)):
            self.values['b2']=np.loadtxt(path+'/b2_%s.txt'%(isUserAutoEncoder), float);
        if os.path.exists(path+'/h_%s.txt'%(isUserAutoEncoder)):
            self.values['h']=np.loadtxt(path+'/h_%s.txt'%(isUserAutoEncoder), float);            
    def saveValues(self,path):
        np.savetxt(path+'/w1_%s.txt'%(isUserAutoEncoder),self.values['w1'],'%.6f');
        np.savetxt(path+'/w2_%s.txt'%(isUserAutoEncoder),self.values['w2'],'%.6f');
        np.savetxt(path+'/b1_%s.txt'%(isUserAutoEncoder),self.values['b1'],'%.6f');
        np.savetxt(path+'/b2_%s.txt'%(isUserAutoEncoder),self.values['b2'],'%.6f');
        np.savetxt(path+'/h_%s.txt'%(isUserAutoEncoder),self.values['h'],'%.6f');    
    def exisValues(self,path):
        if not os.path.exists(path+'/w1_%s.txt'%(isUserAutoEncoder)):
            return False;
        if not os.path.exists(path+'/w2_%s.txt'%(isUserAutoEncoder)):
            return False;        
        if not os.path.exists(path+'/b1_%s.txt'%(isUserAutoEncoder)):
            return False;      
        if not os.path.exists(path+'/b2_%s.txt'%(isUserAutoEncoder)):
            return False;
        if not os.path.exists(path+'/h_%s.txt'%(isUserAutoEncoder)):
            return False;        
        return True;
    def getValues(self):
        return self.values;    
############################ end class ###########################    
    



def actfunc1(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)));


def deactfunc1(x):
    return x*(1.0-x);


def actfunc2(x):
    return x;


def deactfunc2(x):
    return 1;


def check_none(x):
    if x is None:
        return True;
    elif x <= 0.0:
        return True;
    return False;

isUserAutoEncoder=False;
rou=0.1

def run_encoder():
    
    
    tx=7;
    hidden_node=3;
    x = np.array([
            [0.1,0.2,0.3,0.2,0.1,0.2,0.3],
            [0.7,0.6,0.5,0.4,0.3,0.2,0.1]
            ]);
    x=x.T;
    y = np.array([0.7,0.6,0.5,0.4,0.3,0.2,0.1]);
    encoder = BPAutoEncoder(tx,hidden_node,
                            actfunc1,deactfunc1,
                             actfunc2,deactfunc2,check_none);
    encoder.train(x, 0.5, 300);
    
    print(encoder.calculate(y));
    pass;




if __name__ == '__main__':
    run_encoder();
    pass