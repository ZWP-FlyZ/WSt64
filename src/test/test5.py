# -*- coding: utf-8 -*-
'''
Created on 2018年4月12日

@author: zwp
'''

import numpy as np;

import random;
import os;
shape = (339,5825);

a = np.array([[1,2,0,1,0],
              [0,0,0,1,0],
              [0,2,1,1,0],
              [0,0,0,0,0]]);

if __name__ == '__main__':
    print(np.where(a>0));
    t = np.argwhere(a>0);
    print(t);
    for bid,fid in t:
        print(bid,fid);
    print([[] for i in range(4)]);
    
    t = set([1,2,3])-set([2,3]);
    print(random.sample([1,2,3,4,5],4));
    
    print(np.sum(a[[]],axis=0));
    
    print(a[np.where(a>0)]);
     
     
    b = np.array([1,2,3,4]);
    print(np.power(10,b));  
    a[:,1]=0;
    print(a)
        
    pass;