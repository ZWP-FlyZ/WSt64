# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日

@author: zwp12
'''
import numpy as np;

def active_function(x):
    return 1.0/( 1.0 + np.exp(-x));

if __name__ == '__main__':
    
    arr1=np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]]);
    arr2=np.array([1,0,-1]);
    #print(np.mat(arr2)*np.mat(np.reshape(arr1,(3,1))));
    print(np.matmul(arr1,arr2));
    arr1[:,1]=arr1[:,1]-2*arr2;
    print(arr1[:,1]);
    print(active_function(arr1))
    
    print(np.random.normal(0.0,0.6,20));
    
    print(np.sum(arr1[1]*arr2))
    
    pass