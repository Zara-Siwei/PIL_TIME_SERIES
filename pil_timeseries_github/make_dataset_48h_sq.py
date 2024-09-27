# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:52:49 2023

@author: Think book
"""

import scipy.io as spio
import numpy as np
import os
from astropy.io import fits
import datetime

# rootPath = '/data/wangjj/dataset/para_PIL/'
# rootPath = 'F:/dataset/dataset_PIL/'
# savePath = 'F:/dataset/pil/'




# =============================================================================
# 该模块提取了数据的时间序列，并制作了harp和noaa的关系表
# =============================================================================

# rootPath = 'F:/dataset/sharp-events/'
# filePath = rootPath + 'parastru/'
# names = os.listdir(filePath)
# arid = []
# for name in names:
#     strings = name.split('.')
#     if not strings[2] == '6313':
#         arid.append(strings[2])
# arid = np.array(arid).astype('int')

# indexs = np.argsort(arid)
# names_array = np.array(names)
# names_sorted = names_array[indexs]
# arid = arid[indexs]
# # arid.sort()



# tstamp_list = []
# noaa_list = []
# num = 0
# rd = len(arid)
# for i in arid:
#     filefold = rootPath + str(i)
#     file0 = os.listdir(filefold)[0]
#     fits_list = fits.open(filefold+'/'+file0, cache=True)
#     noaa = fits_list[1].header['NOAA_AR']
#     noaa_list.append(noaa)
#     tstamp_list_single = []
#     for file in os.listdir(filefold):
#         if file.endswith("Bp.fits"):
#             string = file.split('.')
#             tstamp = (datetime.datetime.strptime(string[3][0:-4],'%Y%m%d_%H%M%S')).timestamp()
#             tstamp_list_single.extend([tstamp])
#     tstamp_list.append(tstamp_list_single)
#     num += 1
#     if num % (rd // 100) == 0:
#         print('读取进度: {:.0%}'.format(num/rd))

# hnmap = np.zeros((len(arid),2))
# hnmap[:,0] = arid
# hnmap[:,1] = np.array(noaa_list)
# hnmap = hnmap.astype('int')

# tstamp_list = np.array(tstamp_list,dtype=object)
# Path0 = 'F:/dataset/pil/'
# os.makedirs(Path0 + 'info/',exist_ok=True)    

# np.save(Path0 + 'info/'+'tstamp_list.npy', tstamp_list)
# np.save(Path0 + 'info/'+'hnmap.npy', hnmap.astype('int'))



# del tstamp_list

# =============================================================================
# 以上部分只用跑一次
# =============================================================================



# =============================================================================
# 列表信息加载
# =============================================================================
trange_sq = 3
datatype=f"now_{trange_sq}h"
adv_t = [24] 
Path0 = 'H:/dataset/pil/'

tstamp_list = np.load(Path0 + 'info/'+'tstamp_list.npy',allow_pickle=True)
hnmap = np.load(Path0 + 'info/'+'hnmap.npy')


# =============================================================================
# 该模块读取了耀斑列表，用于后续数据分类
# =============================================================================
def h2n(harp):
    """
    to transform the AR num from HARPNUM to NOAANUM
    return a np array
    """
    noaa = hnmap[hnmap[:,0]==harp,1]
    return noaa

import pandas as pd
import datetime

cols=['arnum','level','t0','tm','tt']

file_events = open(Path0 + 'info/flareinfo.txt')
event_list = file_events.readlines()
arnum_list = []
level_list = []
t0_list = []
tm_list = []
tt_list = []

import re
for event in event_list:
    string = re.split('-|T|;|:',event)
    arnum_list.append(string[18][2:])
    level_list.append(string[19][0])
    t0_list.append(datetime.datetime.strptime(string[0]+string[1]+string[2]+string[3]+string[4],'%Y%m%d%H%M'))
    tm_list.append(datetime.datetime.strptime(string[6]+string[7]+string[8]+string[9]+string[10],'%Y%m%d%H%M'))
    tt_list.append(datetime.datetime.strptime(string[12]+string[13]+string[14]+string[15]+string[16],'%Y%m%d%H%M'))

data = pd.DataFrame({'arnum': arnum_list, 'level': level_list, 't0': t0_list, 'tm': tm_list, 'tt': tt_list})
level_list = data['level'].values.tolist()  
dict(zip(*np.unique(level_list, return_counts=True)))
print(data['level'].value_counts())  # 查看英语列各元素出现的次数
index_M = data[data['level']=='M'].index
index_X = data[data['level']=='X'].index
index_M = np.concatenate((index_M,index_X),axis=0)
data_M = data.loc[index_M]
data_M = data_M.reset_index(drop=True)


def classify_M(noaa_input,tstamp_input,N=24,trange=24):
    """
    分类标准描述:
        在训练集中,
        1.所有负例均来自于没有强耀斑的活动区
        2.所有正例均来自于有强耀斑的活动区
        3.时间点后N小时内有M+耀斑发生，记为1
        4.无M+耀斑发生的活动区时间点，记为0
        
    """
    noaa_str = str(noaa_input)
    if noaa_str not in data_M['arnum'].values:
        return 0 # 无M级耀斑的活动区
    else:
        index_AR = data_M[data_M['arnum']==noaa_str].index
    
    t0 = data_M.loc[index_AR]['t0']
    t1 = data_M.loc[index_AR]['tt']
    td = datetime.datetime.fromtimestamp(tstamp_input)
    t0 = t0.reset_index(drop=True)
    t1 = t1.reset_index(drop=True)
    
    # 时间点位于爆发段的前N小时内，记为1
    for i in range(len(t0)):
        if td <= t0[i]:
            if td >= t0[i] - datetime.timedelta(hours=N):
                return 1
            elif td >= t0[i] - datetime.timedelta(hours=N+trange):
                return -1 # 时序需要用24小时连续数据预报未来24小时，因此这类事件可另用
    return -2 # 发生过M级耀斑的活动区，不计0事件，排除



filePath = Path0 + 'parastru/'
for adv in adv_t:
    labeldataPath = Path0+'labeling_PIL_2class_'+datatype+'/'+str(adv)+'/'
    os.makedirs(labeldataPath+'time/',exist_ok=True)
    os.makedirs(labeldataPath+'para/',exist_ok=True)
    os.makedirs(labeldataPath+'label/',exist_ok=True)
    
    num = 0
    rd = len(hnmap)
    # 准备工作结束，以下为数据labeling
    for i in range(len(hnmap)):
    # for i in [0]:
        name = 'hmi.sharp_cea_720s.'+str(hnmap[i,0])+'.sharp_now_parastru.sav'
        file = filePath + name
        s = spio.readsav(file, python_dict=True, verbose=False)
        keys = list(s.keys())
        record_list = list(s['now_parastru0'])
        time_list = []
        para_list = []
        label_list = []
        for rec_ix in range(len(tstamp_list[i])):
            noaa = hnmap[i,1]
            ts = tstamp_list[i][rec_ix]
            label = classify_M(noaa,ts,adv,trange_sq)
            item_para = list(record_list[rec_ix])[1::]
            
            para_list.append(np.array(item_para))
            label_list.extend([label])
            time_list.extend([ts])
            
        np.save(labeldataPath+'time/' + str(hnmap[i,0]) + '.npy', np.array(time_list))
        np.save(labeldataPath+'label/' + str(hnmap[i,0]) + '.npy', np.array(label_list))
        np.save(labeldataPath+'para/' + str(hnmap[i,0]) + '.npy', np.array(para_list))
        num += 1
        if num % (rd // 100) == 0:
            print('标签进度: {:.0%}'.format(num/rd))
    
    
    
    
    
    labeldataPath = Path0+'labeling_PIL_2class_'+datatype+'/'+str(adv)+'/'
    
    datatype1 = 'sq'
    # 数据集制作模块

    import os
    timePath = labeldataPath+'time/'
    paraPath = labeldataPath+'para/'
    labelPath = labeldataPath+'label/'
    

    para0 = np.load(paraPath + str(hnmap[0,0]) + '.npy')
    df_para = pd.DataFrame(para0)
    para_nonan = df_para.dropna(axis=1, how='all')
    #  para_selected = para_nonan.dropna(axis=0, how='all')
    
    # 被保留的参数列
    column_selected = para_nonan.columns
    
    arnum_set = []
    para_set = []
    label_set = []
    time_set = []
    check_nan = []
    index_set = []
    count_from = 0
    if datatype1 == 'sp':
            
        for i in range(len(hnmap)):
            labels = np.load(labelPath + str(hnmap[i,0]) + '.npy')
            paras = np.load(paraPath + str(hnmap[i,0]) + '.npy')
            times = np.load(timePath + str(hnmap[i,0]) + '.npy')
            
            df_paras = pd.DataFrame(paras)
            paras_nonan = df_paras[column_selected]
            paras_selected = paras_nonan.dropna(axis=0, how='any')
            index_selected = paras_selected.index
            label_selected = labels[index_selected]
            time_selected = times[index_selected]
            # data_size.append(len(label_selected))
            # 删除掉无效事件
            index_delete = []
            
            
            
            for j in range(len(label_selected)):
                if label_selected[j] < 0:
                    index_delete.append(j)
                
            
            paras_selected = np.delete(np.array(paras_selected),index_delete,axis=0)
            label_selected = np.delete(np.array(label_selected),index_delete,axis=0)
            time_selected = np.delete(np.array(time_selected),index_delete,axis=0)
            
            
            if len(label_selected) > 0:
                index_set.append([count_from,count_from+len(label_selected)])
                count_from += len(label_selected)
            
                para_set.extend(np.array(paras_selected))
                time_set.extend(np.array(time_selected))
                label_set.extend(label_selected)
                check_nan.append(pd.DataFrame(paras_selected).isnull().values.any())
                arnum_set.append(int(hnmap[i,0]))
                                 
        os.makedirs(Path0 + 'dataset_2class_'+ datatype +'_sp/'+str(adv)+'/',exist_ok=True)
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sp/'+str(adv)+'/para.npy', np.array(para_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sp/'+str(adv)+'/label.npy', np.array(label_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sp/'+str(adv)+'/time.npy', np.array(time_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sp/'+str(adv)+'/countf.npy', np.array(index_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sp/'+str(adv)+'/arnum.npy', np.array(arnum_set))
        if True in check_nan:
            print('Warning: NAN in para!')
    
    if datatype1 == 'sq':
        
        for i in range(len(hnmap)):
            labels = np.load(labelPath + str(hnmap[i,0]) + '.npy')
            paras = np.load(paraPath + str(hnmap[i,0]) + '.npy')
            times = np.load(timePath + str(hnmap[i,0]) + '.npy')
            
            df_paras = pd.DataFrame(paras)
            paras_nonan = df_paras[column_selected]
            paras_selected = paras_nonan.dropna(axis=0, how='any')
            index_selected = paras_selected.index
            label_selected = labels[index_selected]
            time_selected = times[index_selected]
            # data_size.append(len(label_selected))
            # 删除掉无效事件
            index_delete = []
            
            
            
            for j in range(len(label_selected)):
                if label_selected[j] < -1:
                    index_delete.append(j)
                
            
            paras_selected = np.delete(np.array(paras_selected),index_delete,axis=0)
            label_selected = np.delete(np.array(label_selected),index_delete,axis=0)
            time_selected = np.delete(np.array(time_selected),index_delete,axis=0)
            
            
            if len(label_selected) > 0:
                index_set.append([count_from,count_from+len(label_selected)])
                count_from += len(label_selected)
            
                para_set.extend(np.array(paras_selected))
                time_set.extend(np.array(time_selected))
                label_set.extend(label_selected)
                check_nan.append(pd.DataFrame(paras_selected).isnull().values.any())
                arnum_set.append(int(hnmap[i,0]))
                                 
        os.makedirs(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/',exist_ok=True)
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/para.npy', np.array(para_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/label.npy', np.array(label_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/time.npy', np.array(time_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/countf.npy', np.array(index_set))
        np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/arnum.npy', np.array(arnum_set))
        if True in check_nan:
            print('Warning: NAN in para!')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








