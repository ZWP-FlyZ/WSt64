# -*- coding: utf-8 -*-
'''
Created on 2018年3月19日

@author: zwp
'''

from AEwithLocation import Location;


if __name__ == '__main__':
    path = r'/home/zwp/work/Dataset/ws/ws_info.txt';
    lp = Location.LocationProcesser(path);
    di = lp.loc_dict;
    eg=50;
    cot =0;
    cot_s =0;
    for item in di:
        s = len(di[item]);
        if s< eg:
            cot+=1;
            cot_s+=s;
        print('%s:%d'%(item,s));
    print('cot=%d cots=%d'%(cot,cot_s));
    pass