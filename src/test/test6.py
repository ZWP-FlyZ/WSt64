# -*- coding: utf-8 -*-
'''
Created on 2018年6月7日

@author: zwp
'''


import numpy as np;

import random;
import os;



a = np.array([[1,2,0,1,0],
              [0,0,0,1,0],
              [0,2,1,1,0],
              [0,0,0,0,2]]);

b = np.array([
            [ 0 ,1, 2 , 3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 0, 12, 13, 14],
            [15, 16, 17, 18, 19]]);
'''
[[1 2 0 1 0]
 [0 0 0 1 0]
 [0 2 1 1 0]
 [0 0 0 0 2]]
 
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
'''
print(a);
print(b);

ori_contain_all = (a[0]!=0) & (a[2]!=0);
ori_contain_none = (a[0]==0) & (a[2]==0);
ori_coutain_one = np.logical_not(ori_contain_all | ori_contain_none);

print(ori_contain_all);
print(ori_contain_none);
print(ori_coutain_one)

tag_contain_all = (b[0]!=0) & (b[2]!=0);
print(tag_contain_all);

print(np.count_nonzero(ori_contain_all));

delta = np.subtract(b[0],b[2],out=np.zeros_like(b[0]),where=tag_contain_all&ori_contain_all);

print(delta)
if __name__ == '__main__':
    pass