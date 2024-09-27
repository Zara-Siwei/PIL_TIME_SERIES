# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:13:48 2023

@author: Think book
"""
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime

# =============================================================================
# 列表信息加载
# =============================================================================
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


def judge_label(noaa_input,tstamp_input,N=24):
    """
    分类标准描述:
        1.所有负例均来自于没有强耀斑的活动区
        2.所有正例均来自于有强耀斑的活动区
        3.时间点后N小时内有M+耀斑发生，记为1
        4.无M+耀斑发生的活动区时间点，记为0
    """
    noaa_str = str(noaa_input)
    if noaa_str not in data_M['arnum'].values:
        return False # 无M级耀斑的活动区
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
                return True
            # elif td >= t0[i] - datetime.timedelta(hours=N+trange):
            #     return False # 时序需要用24小时连续数据预报未来24小时，因此这类事件可另用
    return False # 发生过M级耀斑的活动区，不计0事件，排除

def judge_label_return_num(noaa_input,tstamp_input,N=24):
    """
    分类标准描述:
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
            # elif td >= t0[i] - datetime.timedelta(hours=N+trange):
            #     return False # 时序需要用24小时连续数据预报未来24小时，因此这类事件可另用
    return -2 # 发生过M级耀斑的活动区，不计0事件，排除

def create_dataset_new(xset,look_back):
    '''
    对数据进行处理，针对数据量较少的1事件
    '''
    dataX = []
    for i in range(len(xset)-look_back+1):
        a = xset[i:(i+look_back),:]
        dataX.append(a)
    Output_X = np.array(dataX)
    return Output_X 

def create_dataset_no_overlap_new(xset,look_back):
    '''
    对数据进行处理，针对数据量较少的1事件
    '''
    dataX = []
    xlen = len(xset)
    for i in range(int(xlen//look_back)):
        a = xset[int(i*look_back):int(i*look_back+look_back),:]
        dataX.append(a)
    Output_X = np.array(dataX)
    return Output_X  



# 第一步，创建不同时间长度的时间序列
t_set1s = [6,12,18,24,30,36,42,48]
for t_set1 in t_set1s:
    
    trange_sq = t_set1
    datatype=f"now_{trange_sq}h"
    # rootPath = '/data/wangjj/dataset/para_PIL/'
    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    adv = 24
    
    para_set = np.load(dataPath  + str(adv)+'/para.npy')
    label_set = np.load(dataPath  + str(adv)+'/label.npy')
    count_from = np.load(dataPath  + str(adv)+'/countf.npy')
    time_set = np.load(dataPath  + str(adv)+'/time.npy')
    arnum_set = np.load(dataPath  + str(adv)+'/arnum.npy')
    
    # para_set = np.squeeze(para_set,axis=1)
    # time_set = []
    # for eacht in time_set_string:
    #     time_set.append(datetime.datetime.strptime(eacht,'%Y.%m.%d_%H:%M:%S'))
    
    # t_length = []
    # for item in count_from:
    #     seconds = time_set[int(item[0]):int(item[1])]
    #     t_length.append( (max(seconds)-min(seconds))/3600 )
    
    # # 不同活动区的数据量分布
    # t_length = []
    # for item in count_from:
    #     hours = item[1]-item[0]
    #     t_length.append(hours)
    
    
    
    # # =============================================================================
    # # 作图分析部分
    # # =============================================================================
    
    # from matplotlib import cm
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # # plt.figure(dpi=300)
    # # plt.subplot(1, 2, 1)
    # fig = plt.figure(dpi=300)
    # ax1 = plt.subplot(1, 1, 1)
    
    # plt.title("Distribution of time range",loc='center',fontproperties="Times New Roman")
    # plt.xlabel('Time range (h)',fontproperties="Times New Roman")
    # plt.ylabel('Number of ARs',fontproperties="Times New Roman")
    # # x=np.array(x1[:,k])
    # # the histogram of the data with histtype='step'
    # n_1, bins1, patches = plt.hist(np.array(t_length), 20,range=(0,max(t_length)),density=False, histtype='stepfilled')
    # plt.setp(patches, 'facecolor', 'red', 'alpha', 0.75)
    
    # plt.setp(patches, 'facecolor', 'dodgerblue', 'alpha', 0.75)
    # plt.rcParams['mathtext.default'] = 'regular'
    # plt.rcParams['font.sans-serif'] = "Times New Roman"
    # plt.rcParams['font.size'] = 12
    # # ax1.set_xlim(-max(max(bins1),max(bins2))*0.05,max(max(bins1),max(bins2))*1.05)
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    
    
    
    
    
    # # =============================================================================
    # # 该模块将全部数据划分为不同的时间序列
    # # 输入数据应按照活动区序号排列
    # # 如果前后两点之间时间不连续，或者处于不同的活动区，则归为不同的时间序列
    # # =============================================================================
    # cf = []  
    # nn=0
    # index_temp = [0,0]
    # for cc in count_from:
    #     index_temp[0] = int(cc[0])
    #     if cc[1]-cc[0] == 1:
    #         index_temp[1] = int(cc[1])
    #         cf.append(np.array(index_temp))
    #         index_temp[0] = int(cc[1])
    #     else:
    #         for aa in range(cc[0],cc[1]):
    #             if aa+1 == cc[1]:
    #                 index_temp[1] = int(cc[1])
    #                 cf.append(np.array(index_temp))
    #                 index_temp[0] = int(cc[1])
    #             else:
    #                 if not time_set[aa+1]-time_set[aa] == 720:
    #                     index_temp[1] = int(aa+1)
    #                     cf.append(np.array(index_temp))
    #                     index_temp[0] = int(aa+1)
    # cf = np.array(cf)
    
            
    # # =============================================================================
    # # 该模块将画出所有时间序列的时长分布，来帮助选择输入参数的步长取值
    # # =============================================================================  
    # t_length1 = []
    # for item in cf:
    #     hours = (item[1]-item[0]-1)*12/60
    #     t_length1.append(hours)
    
    # from matplotlib import cm
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # # plt.figure(dpi=300)
    # # plt.subplot(1, 2, 1)
    # fig = plt.figure(dpi=300)
    # ax1 = plt.subplot(1, 1, 1)
    
    # plt.title("Distribution of time range",loc='center',fontproperties="Times New Roman")
    # plt.xlabel('t (h)',fontproperties="Times New Roman")
    # plt.ylabel('Number of samples',fontproperties="Times New Roman")
    # # x=np.array(x1[:,k])
    # # the histogram of the data with histtype='step'
    # n_1, bins1, patches = plt.hist(np.array(t_length1), 150,range=(0,max(t_length1)),density=False, histtype='stepfilled')
    # plt.setp(patches, 'facecolor', 'red', 'alpha', 0.75)
    
    # plt.setp(patches, 'facecolor', 'dodgerblue', 'alpha', 0.75)
    # plt.rcParams['mathtext.default'] = 'regular'
    # plt.rcParams['font.sans-serif'] = "Times New Roman"
    # plt.rcParams['font.size'] = 12
    # # ax1.set_xlim(-max(max(bins1),max(bins2))*0.05,max(max(bins1),max(bins2))*1.05)
    # plt.grid()
    # plt.yscale('log')
    # plt.tight_layout()
    
    # plt.show()
            
    
    
    # =============================================================================
    # =============================================================================
    
    from scipy.interpolate import make_interp_spline
    
    # label_set = abs(label_set)
    # 针对多维数组的线性差值
    def multinterp(x_time,y_para,x_resamp,k=1):
        y_shape = np.array(y_para.shape)
        if y_shape.ndim == 2:
            y_list = []
            for i in range(len(y_para[0])):
                y_temp = make_interp_spline(x_time,y_para,k=k)(x_resamp)
                y_list.append(y_temp)
            y_array = np.array(y_list)
            return y_array.T
        elif y_shape.ndim == 1:
            return make_interp_spline(x_time,y_para,k=k)(x_resamp)
        else:
            return 'Wrong input!'
    # def create_dataset(xset,yset, look_back):
    #     '''
    #     对数据进行处理，针对数据量较少的1事件
    #     '''
    #     dataX, dataY = [], []
    #     for i in range(len(xset)-look_back+1):
    #         a = xset[i:(i+look_back),:]
    #         dataX.append(a)
    #         dataY.append(yset[i + look_back-1])
    #     Train_X = np.array(dataX)
    #     Train_Y = np.array(dataY)
    
    #     return Train_X, Train_Y


    # def create_dataset_no_overlap(xset,yset, look_back):
    #     '''
    #     对数据进行处理，针对数据量较多的0事件
    #     '''
    #     dataX, dataY = [], []
    #     xlen = len(xset)
    #     for i in range(int(xlen//look_back)):
    #         a = xset[int(i*look_back):int(i*look_back+look_back),:]
    #         dataX.append(a)
    #         dataY.append(yset[int(i*look_back+look_back-1)])
    #     Train_X = np.array(dataX)
    #     Train_Y = np.array(dataY)
    #     return Train_X, Train_Y
    
    # i1 = 286
    # 可见数据多数不连续，需要用插值补全数据并重新切割
    
    para_data = []
    time_data = []
    arnum_data = []
    label_data = []
    
    for i1 in range(len(count_from)):
        
        t_s = time_set[count_from[i1,0]:count_from[i1,1]]
        t_len = max(t_s)-min(t_s)
        

        # 对于时长大于15小时，小于24小时的1事件，插值补全
        if t_len > int(t_set1*5/8)*3600 and t_len < t_set1*3600 and label_set[count_from[i1,1]-1] == 1: # 保留时长大于15h的1事件
            p_s = para_set[count_from[i1,0]:count_from[i1,1]]
            # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
            # seq的label应为最后一个时间点上的label
            l_s = label_set[count_from[i1,1]-1] 
            a_s = arnum_set[i1]
            
            n_fill = int(t_set1*3600/720 + 1 - t_len/(12*60) - 1) 
            p_fill = p_s[0]*np.ones((n_fill,len(p_s[0])))
            t_fill = np.linspace(t_s[0]-(n_fill)*(12*60), t_s[0]-(12*60),n_fill)
            p_c = np.concatenate((p_fill,p_s),axis=0)
            t_c = np.append(t_fill, t_s)
            t_resamp = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
            p_resamp = multinterp(t_c,p_c,t_resamp,k=1)
            
            
            para_data.append(p_resamp)
            time_data.append(t_resamp[-1])
            arnum_data.extend([a_s])
            label_data.extend([l_s])
            

            
        elif t_len >= t_set1*3600:
            # 对于时长大于24小时的1数据，滑动取数据集
            if abs(label_set[count_from[i1,0]]) == 1:
                # 数据滑动间隔为1小时  
                p_s = para_set[count_from[i1,0]:count_from[i1,1]]
                # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
                l_s = label_set[count_from[i1,0]]
                a_s = arnum_set[i1]
                
                t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
                p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
                
                # 剔除1小时的余数
                n_del = (t_len%3600)/720
                p_c = p_resamp1[int(n_del)::]
                t_c = t_resamp1[int(n_del)::]
                
                # 1小时采样分辨率
                t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
                p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
                l_resamp2 = np.ones(len(p_resamp2))*l_s
                p_data = create_dataset_new(p_resamp2,look_back=int(t_set1+1))
                
                # 在对可能包含1类的活动区进行整体插值后，我们需要保证，纳入数据集的
                # 每个序列的末位时间点的判定类型为1，这是为了避免当两个耀斑相隔时间
                # 较长，中间会存在一段应标记为0的区域（实际上不应该取）

                ci = 0
                for p in p_data:
                    t_judge = t_resamp2[int(t_set1+1+ci-1)]
                    if judge_label(int(h2n(a_s)),t_judge,N=24):
                        para_data.append(p)
                        # time_data.append(t_resamp)
                        arnum_data.extend([a_s])
                        label_data.extend([1])
                        time_data.append(t_judge)    
                    ci += 1
                    
            else:
                # 数据滑动间隔为1小时  
                p_s = para_set[count_from[i1,0]:count_from[i1,1]]
                # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
                l_s = label_set[count_from[i1,0]]
                a_s = arnum_set[i1]
                
                # t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
                # p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
                t_resamp1 = np.arange(t_s[0],t_s[-1],3600)
                p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
                
                
                # l_resamp1 = np.ones(len(p_resamp1))*l_s
                p_data = create_dataset_no_overlap_new(p_resamp1,look_back=int(t_set1+1))
                
                ci = 0
    
                for p in p_data:
                    para_data.append(p)
                    # time_data.append(t_resamp)
                    arnum_data.extend([a_s])
                    label_data.extend([l_s])    
                    
                    time_data.append(t_resamp1[int(t_set1+1+ci-1)])    
                    ci += 1
    
    arnum_data_total = list(set(arnum_data))
    print('trange_sq='+str(t_set1))
    print('活动区数目:'+str(len(arnum_data_total)))
    
    
    # 为了保证数据独立性，切分数据集，应该从不同arnum中去选取测试集和训练集
    size_data = []
    ct = 1
    
    
    for ix in range(1,len(arnum_data)):
        if arnum_data[ix] == arnum_data[ix-1]:
            ct += 1
        else:
            size_data.append(ct)
            ct = 1
        # 如果遍历到末位，无法收录当前ar的size数目，需要再收录一次
        if ix == len(arnum_data)-1:
            size_data.append(ct)
    
    countf_data = []
    ct = 0
    for ix in range(len(size_data)):
        cn = [ct,ct+size_data[ix]]
        countf_data.append(cn)
        ct += size_data[ix]
    
    countf_data = np.array(countf_data).astype('int')
    
    import collections
    data_count = collections.Counter(label_data)          
    print('两类样本的数目:\n',data_count)            
    
    
    label_ar = []
    for ii in range(len(countf_data)):
        label_ar.append(label_data[countf_data[ii,0]])
    data_count = collections.Counter(label_ar)
    print('两类活动区的数目:\n',data_count)            
    
    
    np.save(dataPath  + str(adv) + '/para_data.npy',para_data)
    np.save(dataPath  + str(adv) + '/label_data.npy',label_data)
    np.save(dataPath  + str(adv) + '/arnum_data.npy',arnum_data)
    np.save(dataPath  + str(adv) + '/countf_data.npy',countf_data)
    np.save(dataPath  + str(adv) + '/label_ar.npy',label_ar)
    









# =============================================================================
# 找到公有数据集
# =============================================================================

# arnum-label对应表
almap = []
for ixx in range(len(count_from)):
    almap.append([arnum_set[ixx],abs(label_set[count_from[ixx,0]])])
almap = np.array(almap)


adv = 24

trange_sq = 6
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_6h = list(set(ar_d))


trange_sq = 12
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_12h = list(set(ar_d))


trange_sq = 18
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_18h = list(set(ar_d))


trange_sq = 24
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_24h = list(set(ar_d))


trange_sq = 30
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_30h = list(set(ar_d))


trange_sq = 36
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_36h = list(set(ar_d))


trange_sq = 42
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_42h = list(set(ar_d))


trange_sq = 48
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
ar_d = np.load(dataPath  + str(adv) + '/arnum_data.npy')
ar_48h = list(set(ar_d))


ar_shared = set(ar_6h)&set(ar_12h)&set(ar_18h)&set(ar_24h)&set(ar_30h)&set(ar_36h)&set(ar_42h)&set(ar_48h)
ar_shared = list(ar_shared)
ar_shared.sort()

lb_shared = []  
for item in ar_shared:
    index_lb = np.argwhere(almap[:,0]==item)[0][0]
    lb = almap[index_lb,1]
    lb_shared.append(lb)
  
  
print(collections.Counter(lb_shared))











# =============================================================================
# 之前的数据集为较为平衡的数据集，方便神经网络学习到知识
# 而为了评估模型，我们不仅仅需要知道其在大耀斑事件中的判别好不好
# 也需要知道其判别非耀斑事件的能力的好坏，选用真实的不平衡数据集
# =============================================================================

t_set1s = [48]

for t_set1 in t_set1s:
    
    trange_sq = t_set1
    datatype=f"now_{trange_sq}h"
    # rootPath = '/data/wangjj/dataset/para_PIL/'
    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    adv = 24
    
    para_set = np.load(dataPath  + str(adv)+'/para.npy')
    label_set = np.load(dataPath  + str(adv)+'/label.npy')
    count_from = np.load(dataPath  + str(adv)+'/countf.npy')
    time_set = np.load(dataPath  + str(adv)+'/time.npy')
    arnum_set = np.load(dataPath  + str(adv)+'/arnum.npy')
    
    
    
    from scipy.interpolate import make_interp_spline
    
    label_set = abs(label_set)
    # 针对多维数组的线性差值
    def multinterp(x_time,y_para,x_resamp,k=1):
        y_shape = np.array(y_para.shape)
        if y_shape.ndim == 2:
            y_list = []
            for i in range(len(y_para[0])):
                y_temp = make_interp_spline(x_time,y_para,k=k)(x_resamp)
                y_list.append(y_temp)
            y_array = np.array(y_list)
            return y_array.T
        elif y_shape.ndim == 1:
            return make_interp_spline(x_time,y_para,k=k)(x_resamp)
        else:
            return 'Wrong input!'
  
    
    para_data_ub = []
    time_data_ub = []
    arnum_data_ub = []
    label_data_ub = []
    
    for i1 in range(len(count_from)):
        
        t_s = time_set[count_from[i1,0]:count_from[i1,1]]
        t_len = max(t_s)-min(t_s)
        

        # 对于时长大于15小时，小于24小时的1事件，插值补全
        if t_len > int(t_set1*5/8)*3600 and t_len < t_set1*3600 and label_set[count_from[i1,1]-1] == 1: # 保留时长大于15h的1事件
            p_s = para_set[count_from[i1,0]:count_from[i1,1]]
            # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
            # seq的label应为最后一个时间点上的label
            l_s = label_set[count_from[i1,1]-1] 
            a_s = arnum_set[i1]
            
            n_fill = int(t_set1*3600/720 + 1 - t_len/(12*60) - 1) 
            p_fill = p_s[0]*np.ones((n_fill,len(p_s[0])))
            t_fill = np.linspace(t_s[0]-(n_fill)*(12*60), t_s[0]-(12*60),n_fill)
            p_c = np.concatenate((p_fill,p_s),axis=0)
            t_c = np.append(t_fill, t_s)
            t_resamp = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
            p_resamp = multinterp(t_c,p_c,t_resamp,k=1)
            
            
            para_data_ub.append(p_resamp)
            time_data_ub.append(t_resamp[-1])
            arnum_data_ub.extend([a_s])
            label_data_ub.extend([l_s])
            

            
        elif t_len >= t_set1*3600:
            # 对于时长大于24小时的1数据，滑动取数据集
            # 数据滑动间隔为1小时  
            p_s = para_set[count_from[i1,0]:count_from[i1,1]]
            # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
            l_s = label_set[count_from[i1,0]]
            a_s = arnum_set[i1]
            
            t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
            p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
            
            # 剔除1小时的余数
            n_del = (t_len%3600)/720
            p_c = p_resamp1[int(n_del)::]
            t_c = t_resamp1[int(n_del)::]
            
            # 1小时采样分辨率
            t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
            p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
            l_resamp2 = np.ones(len(p_resamp2))*l_s
            p_data = create_dataset_new(p_resamp2,look_back=int(t_set1+1))
            
            # 在对可能包含1类的活动区进行整体插值后，我们需要保证，纳入数据集的
            # 每个序列的末位时间点的判定类型为1，这是为了避免当两个耀斑相隔时间
            # 较长，中间会存在一段应标记为0的区域（实际上不应该取）

            ci = 0
            for p in p_data:
                t_judge = t_resamp2[int(t_set1+1+ci-1)]
                if judge_label_return_num(int(h2n(a_s)),t_judge,N=24)==1:
                    para_data_ub.append(p)
                    # time_data.append(t_resamp)
                    arnum_data_ub.extend([a_s])
                    label_data_ub.extend([1])
                    time_data_ub.append(t_judge)  
                elif judge_label_return_num(int(h2n(a_s)),t_judge,N=24)==0:
                    para_data_ub.append(p)
                    # time_data.append(t_resamp)
                    arnum_data_ub.extend([a_s])
                    label_data_ub.extend([0])
                    time_data_ub.append(t_judge)  
                ci += 1
                    

    
    arnum_data_ub_total = list(set(arnum_data_ub))
    print('trange_sq='+str(t_set1))
    print('活动区数目:'+str(len(arnum_data_ub_total)))
    
    
    # 为了保证数据独立性，切分数据集，应该从不同arnum中去选取测试集和训练集
    size_data = []
    ct = 1
    
    
    for ix in range(1,len(arnum_data_ub)):
        if arnum_data_ub[ix] == arnum_data_ub[ix-1]:
            ct += 1
        else:
            size_data.append(ct)
            ct = 1
        # 如果遍历到末位，无法收录当前ar的size数目，需要再收录一次
        if ix == len(arnum_data_ub)-1:
            size_data.append(ct)
    
    countf_data_ub = []
    ct = 0
    for ix in range(len(size_data)):
        cn = [ct,ct+size_data[ix]]
        countf_data_ub.append(cn)
        ct += size_data[ix]
    
    countf_data_ub = np.array(countf_data_ub).astype('int')
    
    import collections
    data_count = collections.Counter(label_data_ub)          
    print('两类样本的数目:\n',data_count)            
    
    
    label_ar_ub = []
    for ii in range(len(countf_data_ub)):
        label_ar_ub.append(label_data_ub[countf_data_ub[ii,0]])
    data_count = collections.Counter(label_ar_ub)
    print('两类活动区的数目:\n',data_count)            
    
    
    np.save(dataPath  + str(adv) + '/para_data_ub.npy',para_data_ub)
    np.save(dataPath  + str(adv) + '/label_data_ub.npy',label_data_ub)
    np.save(dataPath  + str(adv) + '/arnum_data_ub.npy',arnum_data_ub)
    np.save(dataPath  + str(adv) + '/countf_data_ub.npy',countf_data_ub)
    np.save(dataPath  + str(adv) + '/label_ar_ub.npy',label_ar_ub)






# =============================================================================
# 分折训练
# =============================================================================
# 为了方便比较预报效果，现在需要选取相同的测试集
# 针对八种不同长度的时间序列模型，采用了8种数据规范方法，它们包含有效数据的活动区，
# 这些活动区的公共区域记为AR_shared，至少测试集需要从这些活动区当中进行选择
# 采取的方法为等间隔均匀采样
# =============================================================================

# 设置train/test比例为9/1，每十份里取一份为测试集

each_slice = int(len(ar_shared)/10)
train_rate = 0.9
train_in_slice = int(each_slice*train_rate)
test_in_slice = each_slice-train_in_slice


for fold_selected in [3]:
# fold_selected = 3
    ar_test = []
    for i in range(len(ar_shared)):
        if (i+fold_selected*test_in_slice) % each_slice < test_in_slice:
            ar_test.append(ar_shared[i])
            # ar_test.extend(ar_shared[i])
    ar_test = np.array(ar_test)
    
    # 数据集不需要重新制作，只需要一个总的para_data的文件以及保存不同组合的测试集的arnum
    # 但是每次训练的模型可以按照不同的名称保存
    np.save(Path0 + 'info/' + f'ar_test{fold_selected}.npy',ar_test)
    
    
    
    
    
    # =============================================================================
    # # 训练集数据生成
    # =============================================================================
    trange_sqs = [6,12,18,24,30,36,42,48]
    for trange_sq in trange_sqs:
        datatype=f"now_{trange_sq}h"
        # rootPath = '/data/wangjj/dataset/para_PIL/'
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
        
        
        para_data = np.load(dataPath  + str(adv) + '/para_data.npy')
        label_data = np.load(dataPath  + str(adv) + '/label_data.npy')
        arnum_data = np.load(dataPath  + str(adv) + '/arnum_data.npy')
        countf_data = np.load(dataPath  + str(adv) + '/countf_data.npy')
        label_ar = np.load(dataPath  + str(adv) + '/label_ar.npy')
        
        ar_train = set(arnum_data)-set(ar_test)
        
        para_train = []
        label_train = []
        arnum_train = []
        for i in range(len(countf_data)):
            if arnum_data[countf_data[i,0]] in ar_train:
                para_train.extend(para_data[countf_data[i,0]:countf_data[i,1]])
                label_train.extend(label_data[countf_data[i,0]:countf_data[i,1]])
                arnum_train.extend([arnum_data[countf_data[i,0]]])
        
        para_train = np.array(para_train)
        label_train = np.array(label_train)
        arnum_train = np.array(arnum_train)
        
        np.save(dataPath  + str(adv) + f'/para_train_{fold_selected}.npy',para_train)
        np.save(dataPath  + str(adv) + f'/label_train_{fold_selected}.npy',label_train)
        np.save(dataPath  + str(adv) + f'/arnum_train_{fold_selected}.npy',arnum_train)
    
        
    
    
    # =============================================================================
    # # 如果测试集为平衡数据集
    # =============================================================================
    # 制作48小时测试集数据集，因为其时间长度可以包含更短的时序数据
    trange_sqs = [48]
    for trange_sq in trange_sqs:
        datatype=f"now_{trange_sq}h"
        # rootPath = '/data/wangjj/dataset/para_PIL/'
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
        
        
        para_data = np.load(dataPath  + str(adv) + '/para_data.npy')
        label_data = np.load(dataPath  + str(adv) + '/label_data.npy')
        arnum_data = np.load(dataPath  + str(adv) + '/arnum_data.npy')
        countf_data = np.load(dataPath  + str(adv) + '/countf_data.npy')
        label_ar = np.load(dataPath  + str(adv) + '/label_ar.npy')
        
        ar_train = set(arnum_data)-set(ar_test)
        
        para_test = []
        label_test = []
        arnum_test = []
        for i in range(len(countf_data)):
            if arnum_data[countf_data[i,0]] in ar_test:
                para_test.extend(para_data[countf_data[i,0]:countf_data[i,1]])
                label_test.extend(label_data[countf_data[i,0]:countf_data[i,1]])
                arnum_test.extend([arnum_data[countf_data[i,0]]])
        
        para_test = np.array(para_test)
        label_test = np.array(label_test)
        arnum_test = np.array(arnum_test)
        
        np.save(dataPath  + str(adv) + f'/para_test_{fold_selected}.npy',para_test)
        np.save(dataPath  + str(adv) + f'/label_test_{fold_selected}.npy',label_test)
        np.save(dataPath  + str(adv) + f'/arnum_test_{fold_selected}.npy',arnum_test)
    
    
    # 读取48小时的test数据集
    datatype=f"now_{48}h"
    # rootPath = '/data/wangjj/dataset/para_PIL/'
    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    para_test = np.load(dataPath  + str(adv) + f'/para_test_{fold_selected}.npy')
    label_test = np.load(dataPath  + str(adv) + f'/label_test_{fold_selected}.npy')
    arnum_test = np.load(dataPath  + str(adv) + f'/arnum_test_{fold_selected}.npy')
    # 制作比48小时更短的测试集数据集，可以只取48小时数据集的后部分时间段
    trange_sqs = [6,12,18,24,30,36,42]
    for trange_sq in trange_sqs:
        para_test_now = para_test[:,(48-trange_sq)::,:]
        label_test_now = label_test
        arnum_test_now = arnum_test
        
        datatype = f"now_{trange_sq}h"
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
        
        np.save(dataPath  + str(adv) + f'/para_test_{fold_selected}.npy',para_test_now)
        np.save(dataPath  + str(adv) + f'/label_test_{fold_selected}.npy',label_test_now)
        np.save(dataPath  + str(adv) + f'/arnum_test_{fold_selected}.npy',arnum_test_now)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # =============================================================================
    # # 如果测试集为非平衡数据集
    # =============================================================================
    # 制作48小时测试集数据集，因为其时间长度可以包含更短的时序数据
    trange_sqs = [48]
    for trange_sq in trange_sqs:
        datatype=f"now_{trange_sq}h"
        # rootPath = '/data/wangjj/dataset/para_PIL/'
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
        
        
        para_data = np.load(dataPath  + str(adv) + '/para_data_ub.npy')
        label_data = np.load(dataPath  + str(adv) + '/label_data_ub.npy')
        arnum_data = np.load(dataPath  + str(adv) + '/arnum_data_ub.npy')
        countf_data = np.load(dataPath  + str(adv) + '/countf_data_ub.npy')
        label_ar = np.load(dataPath  + str(adv) + '/label_ar_ub.npy')
        
        ar_train = set(arnum_data)-set(ar_test)
        
        para_test = []
        label_test = []
        arnum_test = []
        countf_test = []
        ct_test = 0
        for i in range(len(countf_data)):
            if arnum_data[countf_data[i,0]] in ar_test:
                para_test.extend(para_data[countf_data[i,0]:countf_data[i,1]])
                label_test.extend(label_data[countf_data[i,0]:countf_data[i,1]])
                arnum_test.extend([arnum_data[countf_data[i,0]]])
                
                datasize = countf_data[i,1]-countf_data[i,0]
                countf_test.append([ct_test,ct_test+datasize])
                ct_test += datasize
                
        para_test = np.array(para_test)
        label_test = np.array(label_test)
        arnum_test = np.array(arnum_test)
        countf_test = np.array(countf_test)
        
        np.save(dataPath  + str(adv) + f'/para_test_ub_{fold_selected}.npy',para_test)
        np.save(dataPath  + str(adv) + f'/label_test_ub_{fold_selected}.npy',label_test)
        np.save(dataPath  + str(adv) + f'/arnum_test_ub_{fold_selected}.npy',arnum_test)
        np.save(dataPath  + str(adv) + f'/countf_test_ub_{fold_selected}.npy',countf_test)
        
    
    # 读取48小时的test数据集
    datatype=f"now_{48}h"
    # rootPath = '/data/wangjj/dataset/para_PIL/'
    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    para_test = np.load(dataPath  + str(adv) + f'/para_test_ub_{fold_selected}.npy')
    label_test = np.load(dataPath  + str(adv) + f'/label_test_ub_{fold_selected}.npy')
    arnum_test = np.load(dataPath  + str(adv) + f'/arnum_test_ub_{fold_selected}.npy')
    countf_test = np.load(dataPath  + str(adv) + f'/countf_test_ub_{fold_selected}.npy')
    
    # 制作比48小时更短的测试集数据集，可以只取48小时数据集的后部分时间段
    trange_sqs = [6,12,18,24,30,36,42]
    for trange_sq in trange_sqs:
        para_test_now = para_test[:,(48-trange_sq)::,:]
        label_test_now = label_test
        arnum_test_now = arnum_test
        countf_test_now = countf_test
        
        datatype = f"now_{trange_sq}h"
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
        
        np.save(dataPath  + str(adv) + f'/para_test_ub_{fold_selected}.npy',para_test_now)
        np.save(dataPath  + str(adv) + f'/label_test_ub_{fold_selected}.npy',label_test_now)
        np.save(dataPath  + str(adv) + f'/arnum_test_ub_{fold_selected}.npy',arnum_test_now)
        np.save(dataPath  + str(adv) + f'/countf_test_ub_{fold_selected}.npy',countf_test_now)
    
    
    





# # 在ub的基础上，又可以制作
# # 制作包含耀斑的活动区数据集
# # arnum-label对应表
# almap = []
# for ixx in range(len(count_from)):
#     almap.append([arnum_set[ixx],abs(label_set[count_from[ixx,0]])])
# almap = np.array(almap)

# trange_sqs = [48]
# for trange_sq in trange_sqs:
#     datatype=f"now_{trange_sq}h"
#     # rootPath = '/data/wangjj/dataset/para_PIL/'
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    
    
#     para_data = np.load(dataPath  + str(adv) + '/para_data_ub.npy')
#     label_data = np.load(dataPath  + str(adv) + '/label_data_ub.npy')
#     arnum_data = np.load(dataPath  + str(adv) + '/arnum_data_ub.npy')
#     countf_data = np.load(dataPath  + str(adv) + '/countf_data_ub.npy')
#     label_ar = np.load(dataPath  + str(adv) + '/label_ar_ub.npy')
    
#     para_test = []
#     label_test = []
#     arnum_test = []
#     countf_test = []
#     ct_test = 0
#     for i in range(len(countf_data)):
#         ar_this = arnum_data[countf_data[i,0]]
#         if ar_this in ar_test:
#             index_lb = np.argwhere(almap[:,0]==ar_this)[0][0]
#             lb_this = almap[index_lb,1]
#             if lb_this == 1:
#                 para_test.extend(para_data[countf_data[i,0]:countf_data[i,1]])
#                 label_test.extend(label_data[countf_data[i,0]:countf_data[i,1]])
#                 arnum_test.extend([arnum_data[countf_data[i,0]]])
                
#                 datasize = countf_data[i,1]-countf_data[i,0]
#                 countf_test.append([ct_test,ct_test+datasize])
#                 ct_test += datasize
            
#     para_test = np.array(para_test)
#     label_test = np.array(label_test)
#     arnum_test = np.array(arnum_test)
#     countf_test = np.array(countf_test)
    
#     np.save(dataPath  + str(adv) + '/para_test_1.npy',para_test)
#     np.save(dataPath  + str(adv) + '/label_test_1.npy',label_test)
#     np.save(dataPath  + str(adv) + '/arnum_test_1.npy',arnum_test)
#     np.save(dataPath  + str(adv) + '/countf_test_1.npy',countf_test)




# # 制作比48小时更短的测试集数据集，可以只取48小时数据集的后部分时间段
# trange_sqs = [6,12,18,24,30,36,42,48]
# for trange_sq in trange_sqs:
#     para_test_now = para_test[:,(48-trange_sq)::,:]
#     label_test_now = label_test
#     arnum_test_now = arnum_test
#     countf_test_now = countf_test
    
#     datatype = f"now_{trange_sq}h"
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    
#     np.save(dataPath  + str(adv) + '/para_test_1.npy',para_test_now)
#     np.save(dataPath  + str(adv) + '/label_test_1.npy',label_test_now)
#     np.save(dataPath  + str(adv) + '/arnum_test_1.npy',arnum_test_now)
#     np.save(dataPath  + str(adv) + '/countf_test_1.npy',countf_test_now)













# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================








# # =============================================================================
# # # 制作包含所有活动区的48小时时长时间序列的非平衡数据集
# # =============================================================================

# trange_sq = 48
# datatype=f"now_{trange_sq}h"
# labeldataPath = Path0+'labeling_PIL_2class_'+datatype+'/'+str(adv)+'/'
# # 数据集制作模块
# import os
# timePath = labeldataPath+'time/'
# paraPath = labeldataPath+'para/'
# labelPath = labeldataPath+'label/'

# # para0 = np.load(paraPath + str(hnmap[0,0]) + '.npy')
# # df_para = pd.DataFrame(para0)
# # para_nonan = df_para.dropna(axis=1, how='all')
# # #  para_selected = para_nonan.dropna(axis=0, how='all')
# import pandas as pd
# # # 被保留的参数列
# # column_selected = para_nonan.columns
# column_selected = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 
#                    14, 15, 16, 17, 18, 19, 20, 23,24]

# arnum_set = []
# para_set = []
# label_set = []
# time_set = []
# check_nan = []
# index_set = []
# count_from = 0

# for i in range(len(hnmap)):
#     labels = np.load(labelPath + str(hnmap[i,0]) + '.npy')
#     paras = np.load(paraPath + str(hnmap[i,0]) + '.npy')
#     times = np.load(timePath + str(hnmap[i,0]) + '.npy')
    
#     df_paras = pd.DataFrame(paras)
#     paras_nonan = df_paras[column_selected]
#     paras_selected = paras_nonan.dropna(axis=0, how='any')
#     index_selected = paras_selected.index
#     label_selected = labels[index_selected]
#     time_selected = times[index_selected]
#     # data_size.append(len(label_selected))
#     # 删除掉无效事件
#     index_delete = []
    
    
    
#     # for j in range(len(label_selected)):
#     #     if label_selected[j] < -999:
#     #         index_delete.append(j)
        
    
#     paras_selected = np.delete(np.array(paras_selected),index_delete,axis=0)
#     label_selected = np.delete(np.array(label_selected),index_delete,axis=0)
#     time_selected = np.delete(np.array(time_selected),index_delete,axis=0)
    
    
#     if len(label_selected) > 0:
#         index_set.append([count_from,count_from+len(label_selected)])
#         count_from += len(label_selected)
    
#         para_set.extend(np.array(paras_selected))
#         time_set.extend(np.array(time_selected))
#         label_set.extend(label_selected)
#         check_nan.append(pd.DataFrame(paras_selected).isnull().values.any())
#         arnum_set.append(int(hnmap[i,0]))
    
    
# para_set = np.array(para_set)
# label_set = np.array(label_set)
# time_set = np.array(time_set)
# arnum_set = np.array(arnum_set)
# count_from = np.array(index_set)

# # os.makedirs(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/',exist_ok=True)
# # np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/para.npy', np.array(para_set))
# # np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/label.npy', np.array(label_set))
# # np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/time.npy', np.array(time_set))
# # np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/countf.npy', np.array(index_set))
# # np.save(Path0 + 'dataset_2class_'+ datatype +'_sq/'+str(adv)+'/arnum.npy', np.array(arnum_set))
# if True in check_nan:
#     print('Warning: NAN in para!')



# t_set1s = [48]

# for t_set1 in t_set1s:
    
#     trange_sq = t_set1
#     datatype=f"now_{trange_sq}h"
#     # rootPath = '/data/wangjj/dataset/para_PIL/'
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
#     adv = 24
    
#     # para_set = np.load(dataPath  + str(adv)+'/para.npy')
#     # label_set = np.load(dataPath  + str(adv)+'/label.npy')
#     # count_from = np.load(dataPath  + str(adv)+'/countf.npy')
#     # time_set = np.load(dataPath  + str(adv)+'/time.npy')
#     # arnum_set = np.load(dataPath  + str(adv)+'/arnum.npy')
    
#     from scipy.interpolate import make_interp_spline
    
#     # label_set = abs(label_set)
#     # 针对多维数组的线性差值
#     def multinterp(x_time,y_para,x_resamp,k=1):
#         y_shape = np.array(y_para.shape)
#         if y_shape.ndim == 2:
#             y_list = []
#             for i in range(len(y_para[0])):
#                 y_temp = make_interp_spline(x_time,y_para,k=k)(x_resamp)
#                 y_list.append(y_temp)
#             y_array = np.array(y_list)
#             return y_array.T
#         elif y_shape.ndim == 1:
#             return make_interp_spline(x_time,y_para,k=k)(x_resamp)
#         else:
#             return 'Wrong input!'
  
    
#     para_data_ub = []
#     time_data_ub = []
#     arnum_data_ub = []
#     label_data_ub = []
    
#     for i1 in range(len(count_from)):
        
#         t_s = time_set[count_from[i1,0]:count_from[i1,1]]
#         t_len = max(t_s)-min(t_s)
        

#         # 对于时长大于15小时，小于24小时的1事件，插值补全
#         if t_len > int(t_set1*5/8)*3600 and t_len < t_set1*3600 and label_set[count_from[i1,1]-1] == 1: # 保留时长大于15h的1事件
#             p_s = para_set[count_from[i1,0]:count_from[i1,1]]
#             # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
#             # seq的label应为最后一个时间点上的label
#             l_s = label_set[count_from[i1,1]-1] 
#             a_s = arnum_set[i1]
            
#             n_fill = int(t_set1*3600/720 + 1 - t_len/(12*60) - 1) 
#             p_fill = p_s[0]*np.ones((n_fill,len(p_s[0])))
#             t_fill = np.linspace(t_s[0]-(n_fill)*(12*60), t_s[0]-(12*60),n_fill)
#             p_c = np.concatenate((p_fill,p_s),axis=0)
#             t_c = np.append(t_fill, t_s)
#             t_resamp = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
#             p_resamp = multinterp(t_c,p_c,t_resamp,k=1)
            
            
#             para_data_ub.append(p_resamp)
#             time_data_ub.append(t_resamp[-1])
#             arnum_data_ub.extend([a_s])
#             label_data_ub.extend([l_s])
            

#         elif t_len >= t_set1*3600:
#             # 对于时长大于24小时的1数据，滑动取数据集
#             # 数据滑动间隔为1小时  
#             p_s = para_set[count_from[i1,0]:count_from[i1,1]]
#             # l_s = label_set[count_from[i1,0]:count_from[i1,1]]
#             l_s = label_set[count_from[i1,0]]
#             a_s = arnum_set[i1]
            
#             t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
#             p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
            
#             # 剔除1小时的余数
#             n_del = (t_len%3600)/720
#             p_c = p_resamp1[int(n_del)::]
#             t_c = t_resamp1[int(n_del)::]
            
#             # 1小时采样分辨率
#             t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
#             p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
#             l_resamp2 = np.ones(len(p_resamp2))*l_s
#             p_data = create_dataset_new(p_resamp2,look_back=int(t_set1+1))
            
#             # 在对可能包含1类的活动区进行整体插值后，我们需要保证，纳入数据集的
#             # 每个序列的末位时间点的判定类型为1，这是为了避免当两个耀斑相隔时间
#             # 较长，中间会存在一段应标记为0的区域（实际上不应该取）

#             ci = 0
#             for p in p_data:
#                 t_judge = t_resamp2[int(t_set1+1+ci-1)]
#                 if judge_label(int(h2n(a_s)),t_judge,N=24):
#                     para_data_ub.append(p)
#                     # time_data.append(t_resamp)
#                     arnum_data_ub.extend([a_s])
#                     label_data_ub.extend([1])
#                     time_data_ub.append(t_judge)  
#                 else:
#                     para_data_ub.append(p)
#                     # time_data.append(t_resamp)
#                     arnum_data_ub.extend([a_s])
#                     label_data_ub.extend([0])
#                     time_data_ub.append(t_judge)  
#                 ci += 1
                    

    
#     arnum_data_ub_total = list(set(arnum_data_ub))
#     print('trange_sq='+str(t_set1))
#     print('活动区数目:'+str(len(arnum_data_ub_total)))
    
    
#     # 为了保证数据独立性，切分数据集，应该从不同arnum中去选取测试集和训练集
#     size_data = []
#     ct = 1
    
    
#     for ix in range(1,len(arnum_data_ub)):
#         if arnum_data_ub[ix] == arnum_data_ub[ix-1]:
#             ct += 1
#         else:
#             size_data.append(ct)
#             ct = 1
#         # 如果遍历到末位，无法收录当前ar的size数目，需要再收录一次
#         if ix == len(arnum_data_ub)-1:
#             size_data.append(ct)
    
#     countf_data_ub = []
#     ct = 0
#     for ix in range(len(size_data)):
#         cn = [ct,ct+size_data[ix]]
#         countf_data_ub.append(cn)
#         ct += size_data[ix]
    
#     countf_data_ub = np.array(countf_data_ub).astype('int')
    
#     import collections
#     data_count = collections.Counter(label_data_ub)          
#     print('两类样本的数目:\n',data_count)            
    
    
#     label_ar_ub = []
#     for ii in range(len(countf_data_ub)):
#         label_ar_ub.append(max(label_data_ub[countf_data_ub[ii,0]:countf_data_ub[ii,1]]))
#     data_count = collections.Counter(label_ar_ub)
#     print('两类活动区的数目:\n',data_count)            
    
    
#     np.save(dataPath  + str(adv) + '/para_data_ub_all.npy',para_data_ub)
#     np.save(dataPath  + str(adv) + '/label_data_ub_all.npy',label_data_ub)
#     np.save(dataPath  + str(adv) + '/arnum_data_ub_all.npy',arnum_data_ub)
#     np.save(dataPath  + str(adv) + '/countf_data_ub_all.npy',countf_data_ub)
#     np.save(dataPath  + str(adv) + '/label_ar_ub_all.npy',label_ar_ub)








# # =============================================================================
# # # 如果测试集为非平衡数据集
# # =============================================================================
# # 制作48小时测试集数据集，因为其时间长度可以包含更短的时序数据
# trange_sqs = [48]
# for trange_sq in trange_sqs:
#     datatype=f"now_{trange_sq}h"
#     # rootPath = '/data/wangjj/dataset/para_PIL/'
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    
    
#     para_data = np.load(dataPath  + str(adv) + '/para_data_ub_all.npy')
#     label_data = np.load(dataPath  + str(adv) + '/label_data_ub_all.npy')
#     arnum_data = np.load(dataPath  + str(adv) + '/arnum_data_ub_all.npy')
#     countf_data = np.load(dataPath  + str(adv) + '/countf_data_ub_all.npy')
#     label_ar = np.load(dataPath  + str(adv) + '/label_ar_ub_all.npy')
    
#     ar_train = set(arnum_data)-set(ar_test)
    
#     para_test = []
#     label_test = []
#     arnum_test = []
#     countf_test = []
#     ct_test = 0
#     for i in range(len(countf_data)):
#         if arnum_data[countf_data[i,0]] in ar_test:
#             para_test.extend(para_data[countf_data[i,0]:countf_data[i,1]])
#             label_test.extend(label_data[countf_data[i,0]:countf_data[i,1]])
#             arnum_test.extend([arnum_data[countf_data[i,0]]])
            
#             datasize = countf_data[i,1]-countf_data[i,0]
#             countf_test.append([ct_test,ct_test+datasize])
#             ct_test += datasize
            
#     para_test = np.array(para_test)
#     label_test = np.array(label_test)
#     arnum_test = np.array(arnum_test)
#     countf_test = np.array(countf_test)
    
#     np.save(dataPath  + str(adv) + '/para_test_ub_all.npy',para_test)
#     np.save(dataPath  + str(adv) + '/label_test_ub_all.npy',label_test)
#     np.save(dataPath  + str(adv) + '/arnum_test_ub_all.npy',arnum_test)
#     np.save(dataPath  + str(adv) + '/countf_test_ub_all.npy',countf_test)
    

# # 读取48小时的test数据集
# datatype=f"now_{48}h"
# # rootPath = '/data/wangjj/dataset/para_PIL/'
# dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
# para_test = np.load(dataPath  + str(adv) + '/para_test_ub_all.npy')
# label_test = np.load(dataPath  + str(adv) + '/label_test_ub_all.npy')
# arnum_test = np.load(dataPath  + str(adv) + '/arnum_test_ub_all.npy')
# countf_test = np.load(dataPath  + str(adv) + '/countf_test_ub_all.npy')

# # 制作比48小时更短的测试集数据集，可以只取48小时数据集的后部分时间段
# trange_sqs = [6,12,18,24,30,36,42]
# for trange_sq in trange_sqs:
#     para_test_now = para_test[:,(48-trange_sq)::,:]
#     label_test_now = label_test
#     arnum_test_now = arnum_test
#     countf_test_now = countf_test
    
#     datatype = f"now_{trange_sq}h"
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    
#     np.save(dataPath  + str(adv) + '/para_test_ub_all.npy',para_test_now)
#     np.save(dataPath  + str(adv) + '/label_test_ub_all.npy',label_test_now)
#     np.save(dataPath  + str(adv) + '/arnum_test_ub_all.npy',arnum_test_now)
#     np.save(dataPath  + str(adv) + '/countf_test_ub_all.npy',countf_test_now)








# # 在ub的基础上，又可以制作
# # 制作包含耀斑的活动区数据集
# # arnum-label对应表
# almap = []
# for ixx in range(len(count_from)):
#     # almap.append([arnum_set[ixx],abs(label_set[count_from[ixx,0]])])
#     almap.append([arnum_set[ixx],max(label_set[count_from[ixx,0]:count_from[ixx,1]])])
    
# almap = np.array(almap)




# trange_sqs = [48]
# for trange_sq in trange_sqs:
#     datatype=f"now_{trange_sq}h"
#     # rootPath = '/data/wangjj/dataset/para_PIL/'
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    
    
#     para_data = np.load(dataPath  + str(adv) + '/para_test_ub_all.npy')
#     label_data = np.load(dataPath  + str(adv) + '/label_data_ub_all.npy')
#     arnum_data = np.load(dataPath  + str(adv) + '/arnum_data_ub_all.npy')
#     countf_data = np.load(dataPath  + str(adv) + '/countf_data_ub_all.npy')
    
#     para_test = []
#     label_test = []
#     arnum_test = []
#     countf_test = []
#     ct_test = 0
#     for i in range(len(countf_data)):
#         ar_this = arnum_data[countf_data[i,0]]
#         if ar_this in ar_test:
#             index_lb = np.argwhere(almap[:,0]==ar_this)[0][0]
#             lb_this = almap[index_lb,1]
#             if lb_this == 1:
#                 para_test.extend(para_data[countf_data[i,0]:countf_data[i,1]])
#                 label_test.extend(label_data[countf_data[i,0]:countf_data[i,1]])
#                 arnum_test.extend([arnum_data[countf_data[i,0]]])
                
#                 datasize = countf_data[i,1]-countf_data[i,0]
#                 countf_test.append([ct_test,ct_test+datasize])
#                 ct_test += datasize
            
#     para_test = np.array(para_test)
#     label_test = np.array(label_test)
#     arnum_test = np.array(arnum_test)
#     countf_test = np.array(countf_test)
    
#     np.save(dataPath  + str(adv) + '/para_test_1.npy',para_test)
#     np.save(dataPath  + str(adv) + '/label_test_1.npy',label_test)
#     np.save(dataPath  + str(adv) + '/arnum_test_1.npy',arnum_test)
#     np.save(dataPath  + str(adv) + '/countf_test_1.npy',countf_test)




# # 制作比48小时更短的测试集数据集，可以只取48小时数据集的后部分时间段
# trange_sqs = [6,12,18,24,30,36,42,48]
# for trange_sq in trange_sqs:
#     para_test_now = para_test[:,(48-trange_sq)::,:]
#     label_test_now = label_test
#     arnum_test_now = arnum_test
#     countf_test_now = countf_test
    
#     datatype = f"now_{trange_sq}h"
#     dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
    
#     np.save(dataPath  + str(adv) + '/para_test_1.npy',para_test_now)
#     np.save(dataPath  + str(adv) + '/label_test_1.npy',label_test_now)
#     np.save(dataPath  + str(adv) + '/arnum_test_1.npy',arnum_test_now)
#     np.save(dataPath  + str(adv) + '/countf_test_1.npy',countf_test_now)





















