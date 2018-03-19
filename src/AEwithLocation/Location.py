# -*- coding: utf-8 -*-
'''
Created on 2018年3月19日

@author: zwp
'''
'''
处理地理位置信息的类和一些操作方法
'''

class LocationProcesser():
    '''
    记录并处理位置信息的类,
    '''
    # 记录总量
    rec_size = 0;
    
    # 包含地理位置数
    rec_loc_size=0;
    
    # id与位置的映射向量
    loc_vector = [];
    
    # 每个地理位置包含id 的映射表
    # {location:[id,id2...]...}
    loc_dict = {}; 
    
    def __init__(self,loc_info_path):
        self.load_data(loc_info_path);
        pass;
    
    def load_data(self,loc_path):
        self.loc_vector=[None];# 置空id为0的地方
        self.loc_dict={};
        self.rec_loc_size=0;
        self.rec_size=0;
        with open(loc_path)  as f:
            for line in f:
                self.rec_size+=1;
                line=line.replace('\n','');
                tid,_,loc = line.split('\t');
                self.loc_vector.append(loc);
                if loc not in  self.loc_dict:
                    self.rec_loc_size+=1;
                    self.loc_dict[loc]=[];
                self.loc_dict[loc].append(int(tid));
    
    
    pass;





