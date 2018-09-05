# -*- coding: utf-8 -*-
'''
Created on 2018年9月5日

@author: zwp12
'''

from tools import utils;


def load_classif(class_path):
    ret = [];
    with open(class_path) as f:
        for line in f:
            ret.append(utils.str2arr(line.strip()));
    return  ret;

def data_split_class(ces,data):
    k = len(ces);
    ret = [[] for _ in range(k)];
    for d in data:
        ci = 0;
        while ci<k:
            if d[1] in ces[ci]:
                break;
            ci+=1;
        if ci<k:
            ret[ci].append(d);
    return ret;

