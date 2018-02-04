# encoding=utf-8
'''
Created on 2018年1月6日

@author: yufangzheng
'''
from LoadData import  LoadData;
from Model import SVD;

if __name__ == "__main__":
    feature = 100
    steps = 30   
    alpha = 0.009
    lambda1 = 0.01
    ld = LoadData()
    filename_train = '/home/zwp/work/Dataset/ws/train/sparseness5/training1.txt' 
    filename_test = '/home/zwp/work/Dataset/ws/test/sparseness5/test1.txt';
    train, test = ld.loadData(filename_train, filename_test)
    model = SVD(train,test,feature,steps,alpha,lambda1)
    model.initialParameter()
    model.learnMF()
    MAE,RMSE = model.calMAEAndRMSE()
    print (MAE,RMSE)