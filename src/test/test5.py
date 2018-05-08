# -*- coding: utf-8 -*-
'''
Created on 2018年4月12日

@author: zwp
'''

import numpy as np;



shape = (339,5825);



if __name__ == '__main__':
    for i in range(1200,2001):
        shape = (339,5825,i);
        N = np.zeros(shape);
        print(i);
        del N;
    pass