# -*- coding: utf-8 -*-
'''
Created on 2018年1月19日

@author: zwp12
'''
import numpy as np;

data_mat = r'E:/Dataset/ws/rtmatrix.txt';

us_shape=(339,5825);

def run_test():
    R = np.loadtxt(data_mat,float);
    sum 
    cot=0;
    for i in range(us_shape[0]):
        for j in range(us_shape[1]):
            if R[i,j]<0:
                R[i,j]=0.0;
            else:
                cot+=1;
    mean = np.sum(R)/cot;
    ek = np.sqrt(np.mean((R-mean)**2));
    print(mean,ek);       
                    
if __name__ == '__main__':
    run_test();
    pass