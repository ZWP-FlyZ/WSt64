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
from tools import SysCheck
from AEwithLocation import BPAE;



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
    def layer_optimize(self,py,y,mask_value=0.0):
        b1 = self.values['b1'];
        w1 = self.values['w1'];
        b2 = self.values['b2'];
        w2 = self.values['w2'];
        h  = self.values['h'];
        lr = self.lr;
        

        #替换py中无效的值为mask_value;
        py = np.where(y!=mask_value,py,y);
        
        origin_w2 = w2.copy();
        # 输出层的调整
        gjs = (py-y)*self.defunc2(py);# 输出层中的梯度
        tmp = gjs*lr;
        b2 = b2 - tmp; # 调整b2
        
        deltaW = np.matmul(
            np.reshape(h,(self.size_hidden,1)),# 隐层输出
            np.reshape(tmp, (1,self.size_x))
            );
        w2 = w2 - deltaW;# 调整w2
        
        tmp = origin_w2* gjs;
        tmp =np.sum(tmp,axis=1); 
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
                        

    def train(self,X,learn_rate,repeat,save_path=None,lr_de=50):
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
            if rep>0 and (rep%lr_de == 0):
                self.lr *= 0.94;
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


def preprocess(R):
    if R is None:
        return R;
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j]<0.0:
                R[i,j] = 0.0;
    #return  (R -  mean) / ek;
    return  R / 20.0; 


# CF预测函数 根据W和S预测出u,s的值,
def predict(u,s,R,W,S):
    a0 = u;
    a1 = s;
    if isICF:
        a0 = s;
        a1 = u;
    sum = 0.0;cot=0.0;
    for item in S[a0,:]:
        if W[a0,item]<=0.0:
            break;
        if R[item,a1] ==NoneValue:
            continue;
        sum+= W[a0,item]*R[item,a1];
        cot+=W[a0,item];
    if cot != 0:
        return sum/cot;
    else:
        return 0.2;


base_path = r'E:';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';


us_shape=(339,5825);
# 是否基于用户的自编码器，预测每个用户的所有服务值
isUserAutoEncoder=True;
# 是否基于服务的CF方法
isICF=False;

# 加载AutoEncoder
loadvalues= False;
continue_train = True;
# 加载相似度矩阵
readWcache=False;

axis0 = us_shape[0];
axis1 = us_shape[1];

if isICF:
    axis0 = us_shape[1];
    axis1 = us_shape[0];

mean = 0.908570086101;
ek = 1.9325920405;
# 训练例子
case = 1;
NoneValue = 0.0;

# autoencoder 参数
hidden_node = 150;
learn_rate=0.07;
repeat = 600;
rou=0.1
test_spa=20;
# 协同过滤参数
k = 13;


# 相似列表，shape=(axis0,k),从大到小
S = None;
R = None;
# 相识度矩阵
W = np.full((axis0,axis0), 0, float);
    

def encoder_run(spa):
    train_data = base_path+'/Dataset/ws/train/sparseness%d/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
       
    values_path=base_path+'/Dataset/ae_values/spa%d'%(spa);
    
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
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    tx = us_shape[0];
    if isUserAutoEncoder:
        tx = us_shape[1];
    encoder = BPAE.BPAutoEncoder(tx,hidden_node,
                            actfunc1,deactfunc1,
                             actfunc1,deactfunc1,check_none);
    if not isUserAutoEncoder:
        R = R.T;
    if loadvalues and encoder.exisValues(values_path):
        encoder.preloadValues(values_path);
    if continue_train:
        encoder.train(R, (learn_rate,100,0.93), repeat,values_path);
        encoder.saveValues(values_path);
    PR = encoder.calFill(R);
    print(R);
    print();
    print(PR);
    print();
############# PR 还原处理   ###############
    PR = PR * 20.0;
    R = R * 20.0;    
    for i in range(PR.shape[0]):
        for j in range(PR.shape[1]):
            if R[i,j]!=NoneValue:
                PR[i,j]=R[i,j];
    print(PR);
    
############# PR 还原处理   ###############        
    if not isUserAutoEncoder:
        PR = PR.T;
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
        rt = PR[int(tc[0]),int(tc[1])];
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= np.sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    

    print('实验结束，总耗时 %.2f秒,稀疏度=%d,MAE=%.6f,RMSE=%.6f\n'%((time.time()-now),spa,mae,rmse));


    print(W)
    print(S)
        
if __name__ == '__main__':
    encoder_run(test_spa);
    pass