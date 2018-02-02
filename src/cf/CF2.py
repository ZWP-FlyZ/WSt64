# -*- coding: utf-8 -*-
'''
Created on 2018年1月16日

@author: zwp12
'''

import numpy as np;
import time;
import tensorflow as tf;
import math;
from math import sqrt
import os;


'''
协同过滤算法，在TensorFlow下的实现，
标志isICF如果为true,则是基于物品的cf;否则为基于用户的CF

一下方法将-1值作为空值，而不是一个状态值

'''


base_path = r'E:/Dataset/wst';
origin_data = base_path+'/rtdata.txt';

train_data = r'E:/Dataset/my/mytest2.txt';
test_data = r'E:/Dataset/my/mytest2.txt';


origin_data = r'E:/Dataset/my/mytest2.txt';



readWcache = False;

# 数据输入形状
isICF = False;
us_shape= (142,4532);
us_shape= (339,5825);

NoneValue = -11;

N1Value = -1;

axis0 = us_shape[0];
axis1 = us_shape[1];
if isICF:
    axis0 = us_shape[1];
    axis1 = us_shape[0];
 
# 相识度矩阵
W = np.full((axis0,axis0), 0, float);

# 相似列表，shape=(axis0,k),从大到小
S = None;
R = None;


sumS=np.zeros(axis0,float);# 平均向量

spas = [5,10,15,20] #稀疏度

case = 1;# 训练与测试例

k=70; #
p = 2 #


# 不同状态的距离
sig = 4.0;

# -1状态判断概率
PN1 = 0.4;




N1cal=[0,#k  k
       0,#-1 k
       0,#k  k
       0];#-1 -1
       
result=[];

# 根据W和S预测出u,s的值,
def predict(u,s):
    global R,W,S,sumS;
    a0 = u;
    a1 = s;
    if isICF:
        a0 = s;
        a1 = u;
    pp = 0.0;sumall=0.0;
    n1sum=0.0;

    for item in S[a0,:]:
        if W[a0,item]<=0.0:
            break;
        if R[item,a1] ==NoneValue:
            continue;
        sumall+=W[a0,item];
        if R[item,a1] ==N1Value:
            n1sum+=W[a0,item];
        else:
            pp+= W[a0,item]*R[item,a1];
        
    if sumall==0:
        return 0.2;
    elif n1sum/sumall >= PN1:
        return -1.0;
    else :
        return pp*1.0/(sumall-n1sum);
    
def run_cf(spa):
    global R,W,S,sumS;
    
    train_data = r'E:/Dataset/ws/train/sparseness%d/training%d.txt'%(spa,case);
    test_data = r'E:/Dataset/ws/test/sparseness%d/test%d.txt'%(spa,case);
    W_path = r'E:/Dataset/ws/W_%s_spa%d_t%d.txt'%(isICF,spa,case);
    
    print('开始实验，isICF=%s,稀疏度=%d,case=%d,sig=%.2f,PN1=%.2f'%(isICF,spa,case,sig,PN1));
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
    if isICF:
        R = R.T;
    # mean = np.mean(R,1);
    del trdata,u,s;
    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    print ('计算相似度矩阵开始');
    tnow = time.time();
    i=0;
    if readWcache and os.path.exists(W_path):
        W = np.loadtxt(W_path, float);
    else:
        for i in range(axis0-1):
            if i%50 ==0:
                print('----->step%d'%(i))
            for j in range(i+1,axis0):
                err = 0.0;
                N = 0;
                for c in range(axis1):
                    if R[i,c]!=NoneValue and R[j,c]!=NoneValue:
                        N+=1;
############################ meth1 #####################################                         
                        if R[i,c]!=N1Value and R[j,c]!=N1Value:
                            err+= abs(R[i,c]-R[j,c])**p;
                        elif R[i,c]==N1Value and R[j,c]==N1Value:
                            continue;
                        else:
                            err += sig**p;
############################# meth1 end ##############################


############################ meth3 #####################################                         
#                         if R[i,c]!=N1Value and R[j,c]!=N1Value:
#                             err+= abs(R[i,c]-R[j,c])**p;
#                         elif R[i,c]==N1Value and R[j,c]==N1Value:
#                             continue;
############################# meth3 end ##############################


##############################  meth2 ##############################
#                        err+= abs(R[i,c]-R[j,c])**p;  #
############################## meth2 end ##############################    
                if N!= 0:
                    # origin W[i,j]=W[j,i]=1.0/(ws ** (1.0/p)+1.0);
                    # W[i,j]=W[j,i]=1.0/( ((ws/cot) ** (1.0/p))+1.0);
                    W[i,j]=W[j,i]= 1.0/math.exp((err/N) ** (1.0/p));
                    # W[i,j]=W[j,i]= 1.0/math.exp(((ws) ** (1.0/p))/cot);
        np.savetxt(W_path,W,'%.6f');                
    print ('计算相似度矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));


    print ('生成相似列表开始');
    tnow = time.time();
    S = np.argsort(-W)[:,0:k];
    for i in range(axis0):
        sumS[i] = np.sum(W[i,S[i]]);            
    print ('生成相似列表开始结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    print ('加载测试数据开始');
    tnow = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));

    print ('评测开始');
    tnow = time.time();
    mae=0.0;rmse=0.0;cot=0;
    n1cal=[0,#k  k
           0,#-1 k
           0,#k  -1
           0];#-1 -1
    for tc in trdata:
        if tc[2]<=0.0:
            continue;
        rt = predict(int(tc[0]),int(tc[1]));
        if rt!=N1Value and tc[2]!=N1Value:
            n1cal[0]+=1;
        elif rt==N1Value and tc[2]!=N1Value:
            n1cal[1]+=1;
        elif rt!=N1Value and tc[2]==N1Value:
            n1cal[2]+=1;
        else:
            n1cal[3]+=1;             
        mae+=abs(rt-tc[2]);
        rmse+=(rt-tc[2])**2;
        cot+=1;
    mae = mae * 1.0 / cot;
    rmse= sqrt(rmse/cot);
    print ('评测完成，耗时 %.2f秒\n'%((time.time() - tnow)));    
    
    print('实验结束，总耗时 %.2f秒,isICF=%s,稀疏度=%d,MAE=%.3f,RMSE=%.3f\n'%((time.time()-now),isICF,spa,mae,rmse));
    print('----------------------------------------------------------\n');    
    for i in range(4):
        N1cal[i]+=n1cal[i];
    print('n1cal',n1cal);
    print('N1cal',N1cal);
    result.append(round(mae,3));
    result.append(round(rmse,3));
    result.append(n1cal);
    print(W);
    print(S);    
if __name__ == '__main__':
    result.append(PN1);
    for spa in spas:
        run_cf(spa);
    result.append(N1cal);
    print('result\n',result);
    pass