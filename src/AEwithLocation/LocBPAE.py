'''
Created on 2018年3月23日

@author: zwp
'''
import os;
import numpy as np;
from AEwithLocation.BPAE import BPAutoEncoder;
from AEwithLocation.Location import LocationProcesser as lp;
class LocAutoEncoder(object):
    '''
    将原数据集按照地理位置分割成若干AE
    '''


    # 正类名 大于等于oeg的类型
    p_all_name='p_all';
    # 反类名 小于oeg的类型
    n_all_name='n_all';
    # 所有类型名
    all_name='all';


    '''
    存储地理位置的index和对应的ae模型
    {loc_name:[
                [index1,index2...],# 该地理位置中包含的序号
                BPAutoEncoder # BP自编码器对象
            ]}
    other为所有反类总和，other2为所有正类总和
    '''
    loc_aes = {p_all_name:[[]],n_all_name:[[]],all_name:[[]]};
    

    # 原始数据集R
    R=None;
    
    isUAE=True;
    
    lp;


    def __init__(self, 
                lp,# LocationProcesser
                oeg,# 将loc数据量小于oeg的值归到’o‘类型中
                R,# 原始数据集[339,5825]
                hidden_size,# 隐层节点数
                func_vec,# 激活函数数组
                isUAE=True# 默认是基于用户的AE
                ):
        '''
            lp：LocationProcesser
            oeg： 将loc数据量小于oeg的值归到’o‘类型中
            R:原始数据集[339,5825]
            hidden_size: 隐层节点数
            func_vec:激活函数数组[actfun1,deactfun1,actfun2,deactfun2]
            isUAE=True: 默认是基于用户的AE
        '''
        self.R = R;
        self.lp = lp;
        self.isUAE = isUAE;
        self.init_ae_loc(lp, oeg, R, hidden_size, func_vec, isUAE);
    
    def init_ae_loc(self,
                    lp,# LocationProcesser
                    oeg,# 将loc数据量小于oeg的值归到’o‘类型中
                    R,# 原始数据集[339,5825]
                    hidden_size,# 隐层节点数
                    func_vec,# 激活函数数组
                    isUAE=True# 默认是基于用户的AE
                    ):
        us_shape=R.shape;
        for k,v in lp.loc_dict.items():
            v = (np.array(v)-1).tolist();
            if len(v)>=oeg:
                self.loc_aes[k]=[v];
                self.loc_aes[self.p_all_name][0].extend(v);
            else:
                self.loc_aes[self.n_all_name][0].extend(v);
            self.loc_aes[self.all_name][0].extend(v);
        self.loc_aes[self.n_all_name][0].sort();
        self.loc_aes[self.p_all_name][0].sort();
        self.loc_aes[self.all_name][0].sort();
        tx=us_shape[0];
        if isUAE: tx=us_shape[1];
        param_vec=[tx,hidden_size,*func_vec,None];
        for _,v in self.loc_aes.items():
            v.append(BPAutoEncoder(*param_vec));
        pass;

    def train_one(self,loc_name,learn_param,repeat,save_path=None,mask_value=0):
        ind = self.loc_aes[loc_name][0];
        encoder = self.loc_aes[loc_name][1];
        X = self.R;
        if not self.isUAE:
            X = self.R.T
        X = X[ind,:];
        print('\n-->训练 %s 的模型开始'%(loc_name));
        encoder.train(X,learn_param, repeat,None,mask_value);
        if save_path != None:
            self.saveValue(save_path,[loc_name]);
        print('-->训练 %s 的模型结束\n'%(loc_name));
        pass;
    
    def train_all(self,learn_param,repeat,save_path=None,mask_value=0):
        loc_list = self.loc_aes.keys();
        print('训练列表：',loc_list);
        for locn in loc_list:
            self.train_one(locn, learn_param, repeat, save_path, mask_value);
        pass;
    
    def train_by_names(self,loc_names,learn_param,repeat,save_path=None,mask_value=0):
        loc_list = loc_names;
        print('训练列表：',loc_list);
        for locn in loc_list:
            self.train_one(locn, learn_param, repeat, save_path, mask_value);
        pass;        
    
    def fill(self,loc_name,R):
        '''
        默认要R为[batch,x_size]
        '''
        return self.loc_aes[loc_name][1].calFill(R);
        
    def getIndexByLocName(self,loc_name):
        return self.loc_aes[loc_name][0];
    
    def saveValue(self,value_path,name_list=None):
        if name_list!=None:
            nl = name_list;
        else:
            nl = self.loc_aes.keys();
        for k in nl:
            npath=value_path+'/'+k;
            if not os.path.isdir(npath):
                os.makedirs(npath);
            v = self.loc_aes[k];
            v[1].saveValues(npath,self.isUAE);
        pass;
    
    def loadValue(self,value_path,name_list=None):
        if name_list!=None:
            nl = name_list;
        else:
            nl = self.loc_aes.keys();
        for k in nl:
            npath=value_path+'/'+k;
            if not os.path.isdir(npath):
                continue;
            v = self.loc_aes[k];
            v[1].preloadValues(npath,self.isUAE);
        pass;
    
    def exitValue(self,value_path,name_list=None):
        if name_list!=None:
            nl = name_list;
        else:
            nl = self.loc_aes.keys();
        for k in nl:
            npath=value_path+'/'+k;
            if not os.path.isdir(npath):
                return False;
        return True;
        pass;
    


        