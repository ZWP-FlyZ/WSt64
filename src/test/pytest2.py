# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日

@author: zwp12
'''
import numpy as np;
def active_function(x):
    return 1.0/( 1.0 + np.exp(-x));

if __name__ == '__main__':

    
    a = np.array([2,0,7,4],dtype=float);
    b = np.array([[1,2,3,4],
                  [6,7,8,9]],dtype=float);
    # print(np.divide(a,a,out=np.zeros_like(a),where=a!=2));
    c = np.array([0,2]);
    print(b[:,c]);
    
    t = np.argwhere(a==-1);
    print(t,a[[1]]);
    
    pass