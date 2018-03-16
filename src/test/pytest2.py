# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日

@author: zwp12
'''
import numpy as np;
def active_function(x):
    return 1.0/( 1.0 + np.exp(-x));

if __name__ == '__main__':
    a = np.array([2,2,7,4])
    b = np.array([[1,0,3,0]])
    print(b!=0);
    print(np.matmul(a,np.reshape(b, (1,4))))
    
    a = np.array([2,0,7,4],dtype=float);
    b = np.array([[1,1,1,1],
                  [2,2,2,2]],dtype=float);
    # print(np.divide(a,a,out=np.zeros_like(a),where=a!=2));
    print(b*a);
    pass