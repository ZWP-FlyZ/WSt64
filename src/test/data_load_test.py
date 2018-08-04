# -*- coding: utf-8 -*-
'''
Created on 2018年7月27日

@author: zwp12
'''

import numpy as np;


base_path = 'E:/work';
origin_data_path = base_path+'/Dataset/wst64/rtdata.txt';

def run():
    shp=(142,4500,64);
    R = np.zeros(shp);
    oridata = np.loadtxt(origin_data_path,float);
    u=np.array(oridata[:,0],int);
    s=np.array(oridata[:,1],int);
    t=np.array(oridata[:,2],int);
    r=np.array(oridata[:,3],float);
    R[u,s,t]=r;
    print(R);

if __name__ == '__main__':
    run();
    pass