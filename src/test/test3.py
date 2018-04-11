# -*- coding: utf-8 -*-
'''
Created on 2018年4月8日

@author: zwp
'''

import numpy as np;
import random;


a = np.reshape(np.arange(50),[5,5,2]);
b = a[[1,2,4],:,:];
c = b[:,[0,3],:];
ac = np.average(c,axis=0);
print(a);
print(b);
print(c)
print(ac);
res = np.sort(ac,axis=0);
print(res);

if __name__ == '__main__':
    
#     for i in range(100):
#         print(random.gauss(0,0.0000001));    
    
    pass