#encoding = utf-8
'''
Created on 2017/5/9

@author: yufangzheng
'''

class LoadData:
    def loadData(self,fileName_train, fileName_test):
        train = {}
        test = {}
        with open(fileName_train) as f:
            for line in f:
                userId, itemId, rating = line.strip().split('\t')
                train.setdefault(int(float(userId))-1, {})
                if float(rating)!= -1:
                    train[int(float(userId))-1][int(float(itemId))-1] = float(rating)
        with open(fileName_test) as f:
            for line in f:
                userId, itemId, rating = line.strip().split('\t')
                test.setdefault(int(float(userId))-1, {})
                if float(rating)!= -1:
                    test[int(float(userId))-1][int(float(itemId))-1] = float(rating)
        return train, test