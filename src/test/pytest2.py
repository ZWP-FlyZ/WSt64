# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日

@author: zwp12
'''
import numpy as np;
def actfunc1(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)));

def testfunc(a,b,c,d):
    print(d,a+b+c);

if __name__ == '__main__':

    
    a = np.array([[[2,0,7,4]]],dtype=float);
    b = np.array([[1,2,2,4],
                  [6,7,8,9],
                  [2,2,2,2],
                  [6,7,8,9]],dtype=float);
    c = np.array([[0,1,1,1],
                  [1,0,0,0]]);
    b[(0,2),:]=c ;             
    print(b)
#     # print(np.divide(a,a,out=np.zeros_like(a),where=a!=2));
#     c = np.array([0,2]);
#     print(b[:,c]);
#     
#     t = np.where(b!=2);
#     print(t,b[[0,1],[1,2]]);
#     
#     
#     d = np.array([[1,2,3,4],
#                [5,6,7,8]]);
#                
#     e = np.array([[1,2,3,4]]);
#     
#     print(d+e);           
#     d = np.reshape(d,(-1,4,1));
#     e = np.reshape(e,(-1,1,4));  
#     print(d,d.shape);
#     print(e,e.shape);
#     
#     ss = d*e;
#     
#     print(np.sum(ss,axis=0));
#     
#     
#     print(actfunc1(a))
    
    para_dict={'a':1, 'b':2, 'c':3, 'd':'l'};
    para_vect=[*[1, 2, 3], 'l'];
    testfunc(*para_vect);
    
    
    
    pass