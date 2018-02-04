# encoding=utf-8
'''
Created on 2017年5月9日

@author: yufangzheng
'''
from __future__ import division 
import copy
from math import sqrt
import random
import numpy as np

class SVD:
    userNum = 339
    itemNum = 5825
    train = {}
    test = {}
    feature = 0
    steps = 0
    alpha = 0
    lambda1 = 0
    p = np.array([])
    q = np.array([])
    bu = np.array([])
    bi = np.array([])
    y = np.array([])
    z = np.array([])
    mean = 0
    
    def __init__(self,train,test,feature,steps,alpha,lambda1):
        self.train = train
        self.test = test
        self.feature = feature
        self.steps = steps
        self.alpha = alpha
        self.lambda1 = lambda1
        self.bu = np.zeros(self.userNum)
        self.bi = np.zeros(self.itemNum)
        self.z = np.zeros((self.userNum,self.feature))
        
    def calMean(self):
        sum = 0
        num = 0
        for u in self.train:
            for i in self.train:
                sum += self.train[u][i]
                num += 1
        self.mean = sum/num
    
    def initialBias(self):
        self.calMean()
        uNum = np.zeros(339)
        iNum = np.zeros(5825)
        for u in self.train:
            for item in self.train[u]:
                iNum[item] += 1
                self.bi[item] += (self.train[u][item] - self.mean)
        for item in self.bi:
            if self.bi[item] != 0:
                self.bi[item] /= (iNum[item] + 25)
        for u in self.train:
            for item in self.train[u]:
                self.bu[u] += (self.train[u][item] - self.mean - self.bi[item])
                uNum[u] += 1
        for u in self.bu:
            if self.bu[u] != 0:
                self.bu[u] /= (uNum[u] + 10)
    
    def initialParameter(self):
        self.p = np.random.rand(self.userNum,self.feature)/sqrt(self.feature)
        self.q = np.random.rand(self.itemNum,self.feature)/sqrt(self.feature)
        self.y = np.random.rand(self.itemNum,self.feature)/sqrt(self.feature)
    
    def learnMF(self):
        for step in range(self.steps):
            print (step)
            RMSE = 0.0
            number = 0
            MAE  = 0.0
            n = 0            
            for u , items in self.train.items():  
                self.z[u] = copy.deepcopy(self.p[u])              
                for i in items:
                    n += 1
                ru = 1/sqrt(n)
                for i in items:
                    self.z[u] += ru*self.y[i]
                
                sum = np.zeros(self.feature)
                for i ,rui in items.items():
                    pui = self.predict(u, i)
                    eui = rui - pui
                    RMSE += pow(eui, 2)
                    MAE += abs(eui)
                    number += 1
                    temp = ru * self.q[i] * eui
                    self.bu[u] += self.alpha * (eui - self.lambda1 * self.bu[u])
                    self.bi[i] += self.alpha * (eui - self.lambda1 * self.bi[i])
                    self.p[u] += self.alpha * (self.q[i] * eui - self.lambda1 * self.p[u])
                    self.q[i] += self.alpha * (self.z[u] * eui - self.lambda1 * self.q[i])
                for i , rui in items.items():
                    self.y[i] += self.alpha * (temp - self.lambda1 * self.y[i])
            RMSE = sqrt(RMSE / number)
            MAE /= number
            print (MAE, RMSE)
            self.alpha *= 0.9
    
    def predict(self,u,i):
        ret = (self.mean + self.bu[int(u)] + self.bi[int(i)]) #聚类均值
        ret += np.dot(self.z[u], self.q[i]) 
        return ret
         
    def predictAll(self):
        #预测
        mean_m = np.zeros((339,5825))
        mean_m.fill(self.mean)
        bu_m = np.mat(self.bu)
        bi_m = np.mat(self.bi)
        bu_m = np.repeat(bu_m, self.itemNum, 0)
        bi_m = np.repeat(bi_m, self.userNum, 0)
        ret = np.dot(self.p,self.q.T)
        ret += mean_m + bu_m.T + bi_m
        print (ret)
            
    def calMAEAndRMSE(self):
        #计算误差
        MAE = 0.0
        RMSE = 0.0
        number = 0
        for u in self.test:
            for i in self.test[u]:
                pui = self.predict(u,i)
                eui = self.test[u][i] - pui
                RMSE += pow(eui, 2)
                MAE += abs(eui)
                number += 1
        RMSE = sqrt(RMSE / number)
        MAE /= number
        print (MAE, RMSE)
        return MAE,RMSE    
            