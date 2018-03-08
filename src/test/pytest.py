# -*- coding: utf-8 -*-
'''
Created on 2018年1月15日

@author: zwp12
'''

import random ;
import numpy as np;
import os;

def randomin(a):
    tmp = random.random()*0.07;
    tmp = a + random.uniform(-tmp,tmp);
    if tmp <0 or tmp>20:
        tmp = 2.0*a-tmp;
    return tmp;

def randnom():
    return abs(random.normalvariate(0.3,0.4));

def randnom2():
    su= random.normalvariate(0,0.5)**2+random.normalvariate(0,1)**2+random.normalvariate(0,1)**2;
    return su;

arr1 = np.array([0,1,2,3]);


if __name__ == '__main__':
    for i in range(100,100):
        print(randnom2());
        

    n = np.array([
                [4,3,2],
                [1,2,3],
                [7,19,2]
                ]);
    

    # print(n);
    k = np.argsort(-n)[:,0:2];
    print(k[2,:]);
    print(np.sum(n[2,k[2]]));
    #print(np.argsort(-n)[:,0:3]);
    
#     W_path = r'E:/Dataset/ws/W_%s_spa%d_t%d.txt'%(False,5,1);
#     
#     print(os.path.exists(W_path))
#     
#     W = np.loadtxt(W_path, float);
#     print(W);
#     print(W.shape);
    
    n1 = np.array([[1,2,3],[4,5,6]]);
    n2 = np.array([[6,5,4],[3,2,1]]);   
    
    print(n1*n2);
    #print(np.matmul(n1,n2));
    
    

    y_sum_types = np.array([50,100]);
    py_sum_types= np.array([55,95]);

    err_type=np.array([20,25]);
    recall=(y_sum_types-err_type)/y_sum_types;
    prec = (y_sum_types-err_type) / py_sum_types;
    macro_f1 = np.mean(2*recall*prec/(recall+prec));
    
    print(recall,prec,macro_f1);
    
    print(np.argwhere(y_sum_types==100));
    
    pass