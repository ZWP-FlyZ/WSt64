# -*- coding: utf-8 -*-
'''
Created on 2018年3月23日

@author: zwp
'''

import os;
import numpy as np;

def removeNoneValue(R):
    '''
    清除无效的数据
    '''
    if R is None:
        return R;
    ind = np.where(R<0);
    R[ind]=0;


if __name__ == '__main__':
    
#     path = r'/home/zwp/work/Dataset/ae_value test';
#     dir = os.path.split(path);
#     print(os.path.isdir(path));
#     os.makedirs(path);
#     print(os.path.isdir(path));
#     print(dir);
    
    for i in range(0,-1,-1):
        print(i);
        
        
    c = [[[1,1,1],1.0,1.1,0],
         [[2,1,1],2.0,2.2,0],
         [[3,1,1],3.0,3.3,0]]    
        
    print(c[0][1:]);
    
    for i in range(10):
        print(np.random.randint(0,10,10));
    
    a = np.array([[1,2,3],[0,1,2]]);
    a[[0,1],[1]*2]=[-1,-2];
    print(a);
    removeNoneValue(a);
    print(a);
    
    
        
    pass