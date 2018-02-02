# -*- coding: utf-8 -*-
'''
Created on 2018年1月19日

@author: zwp12
'''
import numpy as np;

data_mat = r'E:/Dataset/ws/rtmatrix.txt';

us_shape=(339,5825);

def run_test():
    cal = [0,# -1
           0,# 0
           0,# <=0.1
           0,# <=0.2
           0,# <=0.3
           0,# <=0.4
           0,# <=0.5
           0,# <=0.6
           0,# <=0.7
           0,# <=0.8
           0,# <=0.9
           0,# <=1.0
           0,# <=3
           0,# <=5
           0,# <=7
           0,# <=9
           0,# <=11
           0,# <=13
           0,# <=15
           0,# <=17
           0,# <=19
           0,# <20
           0 # 20
           ];
    print(len(cal));
    R = np.loadtxt(data_mat,float);
    all = us_shape[0]*us_shape[1];
    for i in range(us_shape[0]):
        for j in range(us_shape[1]):
            if R[i,j]==-1:cal[0]+=1;
            elif R[i,j]==0:cal[1]+=1;
            elif R[i,j]<=0.1:cal[2]+=1;
            elif R[i,j]<=0.2:cal[3]+=1;
            elif R[i,j]<=0.3:cal[4]+=1;
            elif R[i,j]<=0.4:cal[5]+=1; 
            elif R[i,j]<=0.5:cal[6]+=1;
            elif R[i,j]<=0.6:cal[7]+=1; 
            elif R[i,j]<=0.7:cal[8]+=1;
            elif R[i,j]<=0.8:cal[9]+=1; 
            elif R[i,j]<=0.9:cal[10]+=1;
            elif R[i,j]<=1.0:cal[11]+=1;
            elif R[i,j]<=3:cal[12]+=1; 
            elif R[i,j]<=5:cal[13]+=1;
            elif R[i,j]<=7:cal[14]+=1; 
            elif R[i,j]<=9:cal[15]+=1;
            elif R[i,j]<=11:cal[16]+=1;            
            elif R[i,j]<=13:cal[17]+=1;
            elif R[i,j]<=15:cal[18]+=1;
            elif R[i,j]<=17:cal[19]+=1; 
            elif R[i,j]<=19:cal[20]+=1;
            elif R[i,j]<20:cal[21]+=1; 
            elif R[i,j]==20:cal[22]+=1;
                            
    print('-1=[%d,%.2f]'%(cal[0],cal[0]*1.0/all));        
    print('0=[%d,%.2f]'%(cal[1],cal[1]*1.0/all));
    print('0.1=[%d,%.2f]'%(cal[2],cal[2]*1.0/all));        
    print('0.2=[%d,%.2f]'%(cal[3],cal[3]*1.0/all));
    print('0.3=[%d,%.2f]'%(cal[4],cal[4]*1.0/all));        
    print('0.4=[%d,%.2f]'%(cal[5],cal[5]*1.0/all));
    print('0.5=[%d,%.2f]'%(cal[6],cal[6]*1.0/all));        
    print('0.6=[%d,%.2f]'%(cal[7],cal[7]*1.0/all));
    print('0.7=[%d,%.2f]'%(cal[8],cal[8]*1.0/all));        
    print('0.8=[%d,%.2f]'%(cal[9],cal[9]*1.0/all));
    print('0.9=[%d,%.2f]'%(cal[10],cal[10]*1.0/all));        
    print('1.0=[%d,%.2f]'%(cal[11],cal[11]*1.0/all));
    print('3=[%d,%.2f]'%(cal[12],cal[12]*1.0/all));        
    print('5=[%d,%.2f]'%(cal[13],cal[13]*1.0/all));
    print('7=[%d,%.2f]'%(cal[14],cal[14]*1.0/all));        
    print('9=[%d,%.2f]'%(cal[15],cal[15]*1.0/all));
    print('11=[%d,%.2f]'%(cal[16],cal[16]*1.0/all));        
    print('13=[%d,%.2f]'%(cal[17],cal[17]*1.0/all));
    print('15=[%d,%.2f]'%(cal[18],cal[18]*1.0/all));        
    print('17=[%d,%.2f]'%(cal[19],cal[19]*1.0/all));
    print('19=[%d,%.2f]'%(cal[20],cal[20]*1.0/all));        
    print('<20=[%d,%.2f]'%(cal[21],cal[21]*1.0/all));
    print('20=[%d,%.2f]'%(cal[22],cal[22]*1.0/all));
            
    print('sum=[%d,%d]'%(all,np.sum(np.array(cal)))); 
if __name__ == '__main__':
    run_test();
    pass