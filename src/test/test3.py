# -*- coding: utf-8 -*-
'''
Created on 2018年4月8日

@author: zwp
'''

import numpy as np;
import random;


a = np.reshape(np.arange(25),[5,5]);
b = np.array([0,1,2,3,4]);

print(a);
print(a-b);
c = np.subtract(a,b,out=np.zeros_like(a),where=b!=1);
sc = np.sqrt(np.sum(c**2,axis=1));
ec = np.divide(1,sc,where=sc!=0);
print(c);
print(sc);
print(np.argsort(-ec));

wb = np.argwhere(a>0); 

print(wb);

print(a-b.reshape([-1,1]));

print(np.argsort(b));

print(np.repeat(b.reshape((1,5)), 2, axis=0));

tmp = np.argsort(a)[:,:2]; 
print(tmp);
tmp = tmp.reshape((-1,));
print(tmp)


su = np.zeros((5));

su[tmp]=su[tmp]+1;
print(su);


if __name__ == '__main__':
    
#     for i in range(100):
#         print(random.gauss(0,0.0000001));    
    
    pass