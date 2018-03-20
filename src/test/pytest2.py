# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日

@author: zwp12
'''
import numpy as np;
def actfunc1(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)));

if __name__ == '__main__':

    
    a = np.array([[[2,0,7,4]]],dtype=float);
    b = np.array([[1,2,2,4],
                  [6,7,8,9]],dtype=float);
    # print(np.divide(a,a,out=np.zeros_like(a),where=a!=2));
    c = np.array([0,2]);
    print(b[:,c]);
    
    t = np.argwhere(b!=2);
    print(b[0:1,:]);
    
    
    d = np.array([[1,2,3,4],
               [5,6,7,8]]);
               
    e = np.array([[1,2,3,4]]);
    
    print(d+e);           
    d = np.reshape(d,(-1,4,1));
    e = np.reshape(e,(-1,1,4));  
    print(d,d.shape);
    print(e,e.shape);
    
    ss = d*e;
    
    print(np.sum(ss,axis=0));
    
    
    print(actfunc1(a))
    
    pass