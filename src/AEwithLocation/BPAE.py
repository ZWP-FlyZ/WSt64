# -*- coding: utf-8 -*-
'''
Created on 2018年3月15日

@author: zwp
'''

'''
    由误差逆传播更新参数的自编码器
'''


import numpy as np;
import math;
import time;
import os;

rou = 0.1;

class BPAutoEncoder:
    
    def __init__(self,X_n,hidden_n,actfun1,deactfun1,actfun2,deactfun2,check_none=None):
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
    def calculate(self,x,save_h=True):
        xsp1 = x.shape[0];
        x = np.reshape(x, (xsp1,self.size_x));
        h = self.func1(np.matmul(x,self.values['w1'])+self.values['b1']);
        if save_h:self.values['h']=h;
        y = self.func2(np.matmul(h,self.values['w2'])+self.values['b2']);
        return y;

    def calFill(self,R,x_axis=1):
        '''
        R中与自编码器输入x的x_size对应轴号
        '''
        tR = R;
        if x_axis==0:
            tR = R.T;  
        return self.calculate(tR,False);

    
    def evel(self,py,y,mask_value=0):
        '''
        假定py已经去除mask项
        '''
        index = np.where(y!=mask_value);
        if len(index[0])==0:return 0,0;
        delta = np.abs(py[index]-y[index]);
        mae = np.average(delta);
        rmse = np.average(delta**2);
        return mae,math.sqrt(rmse);
    
    # 更新参数
    def layer_optimize(self,py,y,
                       learn_rate,# 学习速率
                       mask_value=0,
                       err_weight=1.0
                       ):
        b1 = self.values['b1'];
        w1 = self.values['w1'];
        b2 = self.values['b2'];
        w2 = self.values['w2'];
        h  = self.values['h'];
        lr = learn_rate;
        
        #替换py中无效的值为mask_value;
        py = np.where(y!=mask_value,py,y);
        
        origin_w2 = w2.copy();
        # 输出层的调整
        gjs = err_weight*(py-y)*self.defunc2(py);# 输出层中的梯度
        tmp = gjs*lr;
        b2 = b2 - tmp; # 调整b2
        
        deltaW = np.matmul(
            np.reshape(h,(self.size_hidden,1)),# 隐层输出
            np.reshape(tmp, (1,self.size_x))
            );
        w2 = w2 - deltaW;# 调整w2
        
#         tmp = origin_w2* gjs;
#         tmp =np.sum(tmp,axis=1);
#         tmp = np.matmul(gjs,origin_w2.T);
        tmp = np.matmul(gjs,origin_w2.T);
        gis = tmp*self.defunc1(h);
        
        tmp = gis* lr;
        
        b1 = b1 - tmp;# 更新b1

        deltaW = np.matmul(
            np.reshape(y,(self.size_x,1)),# 输入层
            np.reshape(tmp, (1,self.size_hidden))
            );
        w1 = w1 - deltaW;# 调整w1
        

        self.values['b2']=b2;
        self.values['w2']=w2;        
        self.values['b1']=b1;
        self.values['w1']=w1;
                        

    def train(self,X,learn_param,repeat,save_path=None,mask_value=0,weight_list=None):
        '''
        注意输入X为一个矩阵(batch,x_size)
        '''
        self.lp=learn_param;
        lr = learn_param[0];
        de_repeat = learn_param[1];
        de_rate = learn_param[2];
        print('-->训练开始，learn_param=',self.lp,'repeat=%d \n'%(repeat));
        now = time.time();
        shape1=X.shape[0];
        shape2=X.shape[1];
        for rep in range(repeat):
            tnow=time.time();
            maeAll=0.0;rmseAll=0.0;

            for i in range(shape1):
#                 start = i * self.batch;
#                 end = min(start+self.batch,shape2)
#                 x = X[start:end,:];
                x = X[i:i+1,:];
                py = self.calculate(x);

                
                mae,rmse=self.evel(py, x,mask_value);
                err_weight=1.0;
                if weight_list is not None:
                    err_weight = weight_list[i];
                self.layer_optimize(py,x,learn_rate=lr,err_weight=err_weight);
                maeAll+=mae/shape1;
                rmseAll+=rmse/shape1;
#             print(py);
            if rep>0 and rep%de_repeat==0:
                lr*=de_rate;
            if save_path != None:
                self.saveValues(save_path);
            print('---->step%d 耗时%.2f秒 MAE=%.6f RMSE=%.6f'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
        print('\n-->训练结束，总耗时%.2f秒 ,repeat=%d \n'%((time.time()-now),repeat));
 
        
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
        if not os.path.isdir(path):
            os.makedirs(path);
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



