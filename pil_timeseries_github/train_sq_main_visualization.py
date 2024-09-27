# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:47:08 2023

@author: Think book
"""

import numpy as np

"""
    为了选取特定的数据集来进行预报效果的对比
    由于选择了具体的案例，因此需要保证选取同一活动区进行画图
"""

# =============================================================================
# 这部分代码将选出所有M5以上的耀斑数据，以供后续结果的分析
# =============================================================================
datatype = 'now' 
adv_t = [24] 
Path0 = 'H:/dataset/pil/'
tstamp_list = np.load(Path0 + 'info/'+'tstamp_list.npy',allow_pickle=True)
hnmap = np.load(Path0 + 'info/'+'hnmap.npy')

# =============================================================================
# 该模块读取了耀斑总列表，用于后续数据分类
# =============================================================================
def h2n(harp):
    """
    to transform the AR num from HARPNUM to NOAANUM
    return a np array
    """
    noaa = hnmap[hnmap[:,0]==harp,1]
    return noaa

def n2h(noaa):
    """
    to transform the AR num from HARPNUM to NOAANUM
    return a np array
    """
    harp = hnmap[hnmap[:,1]==int(noaa),0]
    return harp

import pandas as pd
import datetime

cols=['arnum','level','t0','tm','tt','harp']

file_events = open(Path0 + 'info/flareinfo.txt')
event_list = file_events.readlines()
arnum_list = []
level_list = []
sublevel_list = []
t0_list = []
tm_list = []
tt_list = []

arharp = []

import re
for event in event_list:
    string = re.split('-|T|;|:',event)
    arnum_list.append(int(string[18][2:]))
    level_list.append(string[19][0])
    sublevel_list.append(float(string[19][1:4]))
    arharp.append(n2h(string[18][2:]))
    t0_list.append(datetime.datetime.strptime(string[0]+string[1]+string[2]+string[3]+string[4],'%Y%m%d%H%M'))
    tm_list.append(datetime.datetime.strptime(string[6]+string[7]+string[8]+string[9]+string[10],'%Y%m%d%H%M'))
    tt_list.append(datetime.datetime.strptime(string[12]+string[13]+string[14]+string[15]+string[16],'%Y%m%d%H%M'))

data = pd.DataFrame({'arnum': arnum_list, 'level': level_list, 'sublevel':sublevel_list, 't0': t0_list, 'tm': tm_list, 'tt': tt_list, 'harp':arharp})
level_list = data['level'].values.tolist()  
dict(zip(*np.unique(level_list, return_counts=True)))    
print(data['level'].value_counts())  # 查看英语列各元素出现的次数
index_M = data[data['level']=='M'].index
index_X = data[data['level']=='X'].index
index_M = np.concatenate((index_M,index_X),axis=0)
data_M = data.loc[index_M]
data_M = data_M.reset_index(drop=True)



# =============================================================================
# 输出M5级别以上的耀斑列表
# =============================================================================
# arnum_train = []
ar_num = []
ar_lv = []
ar_sublv = []

t0_list = []
tm_list = []
tt_list = []
# 将大于M5的耀斑挑出来
for ii in range(len(data)):
    if data['level'][ii] == 'M':
        if data['sublevel'][ii] >= 5:
            ar_num.append(data['arnum'][ii])
            ar_lv.append(data['level'][ii])
            ar_sublv.append(data['sublevel'][ii])
            t0_list.append(data['t0'][ii])
            tm_list.append(data['tm'][ii])
            tt_list.append(data['tt'][ii])
    elif data['level'][ii] == 'X':
        ar_num.append(data['arnum'][ii])
        ar_lv.append(data['level'][ii])
        ar_sublv.append(data['sublevel'][ii])
        t0_list.append(data['t0'][ii])
        tm_list.append(data['tm'][ii])
        tt_list.append(data['tt'][ii])
        

data_M5 = pd.DataFrame({'arnum':ar_num,'level':ar_lv,'sublevel':ar_sublv,'t0':t0_list,'tm':tm_list,'tt':tt_list})    
dict(zip(*np.unique(ar_num, return_counts=True)))    
print(data_M5['level'].value_counts())  # 查看英语列各元素出现的次数


# =============================================================================
# 输出C7级别以上的耀斑列表
# =============================================================================
# arnum_train = []
ar_num = []
ar_lv = []
ar_sublv = []

t0_list = []
tm_list = []
tt_list = []
# 将大于M5的耀斑挑出来
for ii in range(len(data)):
    if data['level'][ii] == 'C':
        if data['sublevel'][ii] >= 7:
            ar_num.append(data['arnum'][ii])
            ar_lv.append(data['level'][ii])
            ar_sublv.append(data['sublevel'][ii])
            t0_list.append(data['t0'][ii])
            tm_list.append(data['tm'][ii])
            tt_list.append(data['tt'][ii])
    elif data['level'][ii] == 'M':
        ar_num.append(data['arnum'][ii])
        ar_lv.append(data['level'][ii])
        ar_sublv.append(data['sublevel'][ii])
        t0_list.append(data['t0'][ii])
        tm_list.append(data['tm'][ii])
        tt_list.append(data['tt'][ii])
    elif data['level'][ii] == 'X':
        ar_num.append(data['arnum'][ii])
        ar_lv.append(data['level'][ii])
        ar_sublv.append(data['sublevel'][ii])
        t0_list.append(data['t0'][ii])
        tm_list.append(data['tm'][ii])
        tt_list.append(data['tt'][ii])
        

data_C7 = pd.DataFrame({'arnum':ar_num,'level':ar_lv,'sublevel':ar_sublv,'t0':t0_list,'tm':tm_list,'tt':tt_list})    
dict(zip(*np.unique(ar_num, return_counts=True)))    
print(data_C7['level'].value_counts())  # 查看英语列各元素出现的次数




trange_sq = 48
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
adv = 24
arnum_test_48h = np.load(dataPath  + str(adv) + '/arnum_test.npy')
arnum_train_48h = np.load(dataPath  + str(adv) + '/arnum_train.npy')



trange_sq = 36
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
adv = 24
arnum_test_36h = np.load(dataPath  + str(adv) + '/arnum_test.npy')
arnum_train_36h = np.load(dataPath  + str(adv) + '/arnum_train.npy')


trange_sq = 24
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
adv = 24
arnum_test_24h = np.load(dataPath  + str(adv) + '/arnum_test.npy')
arnum_train_24h = np.load(dataPath  + str(adv) + '/arnum_train.npy')


trange_sq = 12
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
adv = 24
arnum_test_12h = np.load(dataPath  + str(adv) + '/arnum_test.npy')
arnum_train_12h = np.load(dataPath  + str(adv) + '/arnum_train.npy')


trange_sq = 6
datatype=f"now_{trange_sq}h"
# rootPath = '/data/wangjj/dataset/para_PIL/'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
adv = 24
arnum_test_6h = np.load(dataPath  + str(adv) + '/arnum_test.npy')
arnum_train_6h = np.load(dataPath  + str(adv) + '/arnum_train.npy')


arnum_train_shared = list(set(arnum_train_24h) & set(arnum_train_48h) & set(arnum_train_12h) & set(arnum_train_6h) & set(arnum_train_36h))
arnum_test_shared = list(set(arnum_test_24h) & set(arnum_test_48h) & set(arnum_test_12h) & set(arnum_test_6h) & set(arnum_test_36h))

# np.save('H:/dataset/pil/' + '/arnum_test_shared.npy',arnum_test_shared)
# np.save('H:/dataset/pil/' + '/arnum_train_shared.npy',arnum_train_shared)



# =============================================================================
# 分别从训练集和测试集当中，选出大耀斑活动区，即C7及以上耀斑的活动区
# =============================================================================

# 训练集中的大耀斑活动区
arnum_test_noaa = [h2n(x) for x in arnum_test_shared]
arnum_test_noaa = np.array(arnum_test_noaa).reshape((len(arnum_test_noaa),))
list_arnum_test_noaa = list(set(arnum_test_noaa))
arnum_test_noaa = np.array(list_arnum_test_noaa)
arnum_test_noaa.sort() 
arnum_test_chosen = []
for item in arnum_test_noaa:
    if item in list(data_C7['arnum']):
        arnum_test_chosen.append(item)

print(f"在测试集中一共查到{len(arnum_test_chosen)}个共同活动区")

# 训练集中的大耀斑活动区
arnum_train_noaa = [h2n(x) for x in arnum_train_shared]
arnum_train_noaa = np.array(arnum_train_noaa).reshape((len(arnum_train_noaa),))
list_arnum_train_noaa = list(set(arnum_train_noaa))
arnum_train_noaa = np.array(list_arnum_train_noaa)
arnum_train_noaa.sort() 
arnum_train_chosen = []
for item in arnum_train_noaa:
    if item in list(data_C7['arnum']):
        arnum_train_chosen.append(item)

print(f"在训练集中一共查到{len(arnum_train_chosen)}个共同活动区")



# =============================================================================
# 设置部分
# =============================================================================



INPUT_DIMS = 18
TIME_STEPS = 24 + 1
lstm_units = 64



# adv = 24
# fold_selected = 2
# trange_sq = 24
# # 读取48小时的test数据集
# datatype=f"now_{trange_sq}h"
# # rootPath = '/data/wangjj/dataset/para_PIL/'
# dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'


# para_train = np.load(dataPath + str(adv)+f'/para_train_{fold_selected}.npy')
# label_train = np.load(dataPath + str(adv)+f'/label_train_{fold_selected}.npy')
# para_test = np.load(dataPath + str(adv)+f'/para_test_{fold_selected}.npy')
# label_test = np.load(dataPath + str(adv)+f'/label_test_{fold_selected}.npy')
# arnum_test = np.load(dataPath + str(adv)+f'/arnum_test_{fold_selected}.npy',allow_pickle=True)
# arnum_train = np.load(dataPath + str(adv)+f'/arnum_train_{fold_selected}.npy',allow_pickle=True)




datatype = 'now'
dataPath = 'H:/dataset/pil/' + 'dataset_2class_' + datatype + '_sq/'
adv = 24
para_train = np.load(dataPath + str(adv)+'/para_train.npy')
label_train = np.load(dataPath + str(adv)+'/label_train.npy')
para_test = np.load(dataPath + str(adv)+'/para_test.npy')
label_test = np.load(dataPath + str(adv)+'/label_test.npy')
arnum_test = np.load(dataPath + str(adv)+'/arnum_test.npy',allow_pickle=True)
arnum_train = np.load(dataPath + str(adv)+'/arnum_train.npy',allow_pickle=True)

label_test = abs(label_test)
label_train = abs(label_train)

para_2d_train = para_train.reshape((para_train.shape[0]*para_train.shape[1],para_train.shape[2]))
para_2d_test = para_test.reshape((para_test.shape[0]*para_test.shape[1],para_test.shape[2]))

#归一化
# print(para_train.shape) 
from sklearn.preprocessing import MinMaxScaler
scalarX = MinMaxScaler()
scalarX.fit(para_2d_train)
# scalarY.fit(y_train)
x_train = scalarX.transform(para_2d_train)
# y = scalarY.transform(y_train)

x_test = scalarX.transform(para_2d_test)

x_train = x_train[:,2::]
x_test = x_test[:,2::]

x_train = x_train.reshape((para_train.shape[0],para_train.shape[1],x_train.shape[1]))
x_test = x_test.reshape((para_test.shape[0],para_test.shape[1],x_test.shape[1]))

# pollution_data = data[:,0].reshape(len(data),1)


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(label_train)
encoded_Ytest = encoder.transform(label_test)

# convert integers to dummy variables (one hot encoding)

dummy_y = np_utils.to_categorical(encoded_Y)
# dummy_ytest = np_utils.to_categorical(encoded_Ytest)
 
uniques, ids = np.unique(encoded_Y, return_inverse=True)




# import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


model_pre = dataPath  + str(adv) + '/check_100_0.1489_cnn-lstm-at_now_24h_fold2.hdf5'
m = load_model(dataPath  + str(adv) + '/check_10_0.2150_cnn-lstm-at-now.hdf5')
# m = load_model('H:/dataset/pil/dataset_2class_sharp_sq/24\check_190_0.0264_cnn-lstm-at-sharp-new0.hdf5')
# m = load_model(model_pre)
m.summary()

pred = m.predict(x_test)
pred_01 = uniques[pred.argmax(1)]
y_pred = encoder.inverse_transform(pred_01)


from sklearn.metrics import classification_report  
res = classification_report(label_test,y_pred,labels=[0,1],target_names=['非强耀斑','强耀斑'])
print(str(adv) + '小时的提前量下，预报的结果:')
print(res)

# 输入原数据，经过相同的归一化操作







# 选测试集下
ix = 286
### arnum = 1946
sample = x_test[ix].reshape((1,x_test[ix].shape[0],x_test[ix].shape[1]))

# =============================================================================
# 可视化第一部分，画出各个层的特征图
# =============================================================================
import matplotlib.pyplot as plt
from keras import models
# m = load_model(dataPath  + str(adv) + '/check_10_0.2150_cnn-lstm-at-now.hdf5')
m = load_model(dataPath  + str(adv) + '/check_10_0.2132_cnn-lstm-at-ppt.hdf5')

# layer_outputs = [layer.output for layer in m.layers[:4]]
# 这一步很关键，跳过输入层
layer_outputs = [layer.output for layer in m.layers][1:]
# 
activation_model = models.Model(inputs=m.input, outputs=layer_outputs)
# 获得改样本的特征图
activations = activation_model.predict(sample)
# 显示第一层激活输出特的第一个滤波器的特征图
first_layer_activation = activations[6]
plt.matshow(first_layer_activation[0,:,:],  cmap="seismic")


# 存储层的名称
layer_names = []
for layer in m.layers[1:9]:
    layer_names.append(layer.name)
# 每行显示16个特征图
images_pre_row = 8  # 每行显示的特征图数
# 循环8次显示8层的全部特征图
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1] #保存当前层的特征图个数
    size = layer_activation.shape[1]  #保存当前层特征图的宽高
    n_col = n_features // images_pre_row #计算当前层显示多少行
    # 生成显示图像的矩阵
    display_grid = np.zeros((size*n_col, images_pre_row*size))
    # 遍历将每个特张图的数据写入到显示图像的矩阵中
    for col in range(n_col):
        for row in range(images_pre_row):
            #保存该张特征图的矩阵(size,size,1)
            channel_image = layer_activation[0,:,:,col*images_pre_row+row]
            #为使图像显示更鲜明，作一些特征处理
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            #把该特征图矩阵中不在0-255的元素值修改至0-255
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            #该特征图矩阵填充至显示图像的矩阵中
            display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_image
    scale = 1./size
    #设置该层显示图像的宽高
    plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    #显示图像
    plt.imshow(display_grid, aspect="auto", cmap="viridis")


# =============================================================================
# 可视化第二部分，用热力矩阵图表征哪部分的特征更重要
# =============================================================================
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import layers
from keras import models
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

m = load_model(dataPath  + str(adv) + '/check_10_0.2150_cnn-lstm-at-now.hdf5')

m.summary()
output_1 = m.output[:, 1]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = m.get_layer('attention_mul')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(output_1, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1))
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
# pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([m.input], [pooled_grads, last_conv_layer.output[0]])

x = sample
pooled_grads_value, conv_layer_output_value = iterate(x)
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(128):
    conv_layer_output_value[:,i] *= pooled_grads_value[i]
# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

from matplotlib.dates import  DateFormatter

# 基本图例绘图
import matplotlib.pyplot as plt
xa = np.arange(-24,1,1)
a = heatmap

fig=plt.figure(dpi=300,figsize=(12,5))
ax1=fig.add_subplot(111)

key1=ax1.plot(xa, a ,'orange', linewidth = 3,linestyle ='-', label = 'sequence sample')

fonttype = 'Times New Roman'
labelsize = 20
sticksize = 20

unit=' (h)'
ax1.set_title("heatmap strength",size=20,fontproperties=fonttype)
ax1.set_xlabel("time advance"+unit,size=labelsize,fontproperties=fonttype)
ax1.set_ylabel("importance",size=labelsize,fontproperties=fonttype)

ax1.grid(ls="-.")
plt.xticks(fontsize=sticksize,fontproperties=fonttype)
plt.yticks(fontsize=sticksize,fontproperties=fonttype) 
# ax1.set_yticks(ticks =default,fontsize=10,fontproperties=fonttype) 
plt.rcParams['mathtext.default'] = 'regular'

ax1.grid(ls="-.")
# ax2.grid(ls="--")
font = {'family':'Times New Roman'  #'serif', 
         #         ,'style':'italic'
        ,'weight':'normal'
         #         ,'color':'red'
        ,'size':20
       }
key = key1
labs = [l.get_label() for l in key]
ax1.legend(key,labs,prop=font
           ,loc = 'upper right' 
           ,bbox_to_anchor=(0.99, 0.99)   # (x, y, width, height) (0, 0.5, 0.5, 0.5)
           # , bbox_to_anchor=(0.5, 0.5, 0.5, 0.5)
           , markerscale = 1.5 # legend里面的符号的大小
           )

plt.show()






# 基本图例绘图
import matplotlib.pyplot as plt
colnum = 0
colnames = ['TOTUSJH','TOTPOT','TOTUSJZ','ABSNJZH','SAVNCPP',
            'USFLUX','AREA_ACR','MEANPOT','R_VALUE','SHRGT45',
            'MEANSHR','MEANGAM','MEANGBT','MEANGBH','MEANGBZ',
            'MEANJZH_abs','MEANJZD_abs','MEANALP_abs']

xa = np.arange(-24,1,1)
a = x[0,:,colnum]

fig=plt.figure(dpi=300,figsize=(12,5))
ax1=fig.add_subplot(111)

key1=ax1.plot(xa, a ,'orange', linewidth = 3,linestyle ='-', label = 'sequence sample')

fonttype = 'Times New Roman'
labelsize = 20
sticksize = 20

unit=' (h)'
ax1.set_title("test sample "+str(ix),size=20,fontproperties=fonttype)
ax1.set_xlabel("time advance"+unit,size=labelsize,fontproperties=fonttype)
ax1.set_ylabel(colnames[colnum],size=labelsize,fontproperties=fonttype)

ax1.grid(ls="-.")
plt.xticks(fontsize=sticksize,fontproperties=fonttype)
plt.yticks(fontsize=sticksize,fontproperties=fonttype) 
# ax1.set_yticks(ticks =default,fontsize=10,fontproperties=fonttype) 
plt.rcParams['mathtext.default'] = 'regular'

ax1.grid(ls="-.")
# ax2.grid(ls="--")
font = {'family':'Times New Roman'  #'serif', 
         #         ,'style':'italic'
        ,'weight':'normal'
         #         ,'color':'red'
        ,'size':20
       }
key = key1
labs = [l.get_label() for l in key]
ax1.legend(key,labs,prop=font
           ,loc = 'upper right' 
           ,bbox_to_anchor=(0.99, 0.99)   # (x, y, width, height) (0, 0.5, 0.5, 0.5)
           # , bbox_to_anchor=(0.5, 0.5, 0.5, 0.5)
           , markerscale = 1.5 # legend里面的符号的大小
           )

plt.show()


import scipy.io as spio
import matplotlib.pyplot as plt
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

from scipy.interpolate import make_interp_spline


def stp2t(arr_tstamp):
    arr_time = []
    for item in arr_tstamp:
        arr_time.append(datetime.datetime.fromtimestamp(item))
    return np.array(arr_time)


def para_plot(arnum_noaa,time_range=0):
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    
    
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    

    t_s = time_selected
    p_s = paras_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    p_c = p_resamp1[int(n_del)::]
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
    
    t_plot = t_resamp2
    c_plot = 2
    p_plot = p_resamp2[:,c_plot]
    
    
    # l_resamp2 = np.ones(len(p_resamp2))*1
    # p_data,_ = create_dataset(p_resamp2,l_resamp2, look_back=trange_sq+1)
    

    
    
    fig=plt.figure(figsize=(8,5),dpi=300)
    ax1=fig.add_subplot(111)
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    font = {'family':'Times New Roman','weight':'normal','size':15}
    ax1.xaxis.set_major_formatter(DateFormatter('%d')) # '%Y%m%d%H%M'
    ax1.set_xlabel('day',fontsize=20,fontproperties="Times New Roman")
    ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$',fontsize=20,fontproperties="Times New Roman")
    
    t_max = max(stp2t(t_plot))
    t_min = min(stp2t(t_plot))
    plt.title('NOAA'+str(arnum_noaa)+'\n'+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
              fontproperties="Times New Roman",fontsize=20)
    h1 = ax1.plot(stp2t(t_plot),p_plot,'royalblue',label='para',linestyle='--',marker='.',markersize=8)
    # ixs = 0
    # for each_t0 in list(t0s_C):
    #     h3 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],color='gold',linestyle='-.',label='$t_0$ of C-class flare')
    #     # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
    #     ixs += 1
    # for each_t0 in list(t0s_M):
    #     h2 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='$t_0$ of M/X-class flare')
    #     # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
    #     ixs += 1
    plt.xticks(fontsize=20,fontproperties="Times New Roman")
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    ax2 = ax1.twinx() # 包含另一个y轴的坐标轴对象
    # h4 = ax2.plot(pred_t,prob_1,'g',label='pred')
    ax2.set_ylabel('probability',fontsize=20,fontproperties="Times New Roman")
    # ax2.set_xlim(0,1)
    ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax2.grid(linestyle="--",axis='y')
    # if len(t0s_M) > 0:
    #     if len(t0s_C) > 0:
    #         keys = h1 + h2 + h3 + h4
    #     else:
    #         keys = h1 + h2 + h4
    # elif len(t0s_C) > 0:
    #     keys = h1 + h3 + h4
    # else:
    #     keys = h1 + h4
    keys = h1
    labs = [l.get_label() for l in keys]
    legend = ax1.legend(keys,labs,prop=font,loc = 'upper left',bbox_to_anchor=(0.01, 0.99))
    handles = legend.legendHandles
    for i in range(len(handles)):
        # label = handles[i].get_label()
        handles[i].set_alpha(0.5)
    
    if time_range == 0:
        ax1.set_xlim(t_min-(t_max-t_min)*0.05,t_max+(t_max-t_min)*0.05)
    else:
        ax1.set_xlim(time_range[0]-(time_range[1]-time_range[0])*0.05,time_range[1]+(time_range[1]-time_range[0])*0.05)
    plt.xticks(fontsize=20,fontproperties="Times New Roman")
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    plt.tight_layout()
    plt.show()







para_plot(h2n(1946)[0])
























import numpy as np
# =============================================================================
# 这部分代码将选出所有M5以上的耀斑数据，以供后续结果的分析
# =============================================================================
datatype = 'now' 
adv_t = [24] 
Path0 = 'H:/dataset/pil/'
tstamp_list = np.load(Path0 + 'info/'+'tstamp_list.npy',allow_pickle=True)
hnmap = np.load(Path0 + 'info/'+'hnmap.npy')

# =============================================================================
# 该模块读取了耀斑总列表，用于后续数据分类
# =============================================================================
def h2n(harp):
    """
    to transform the AR num from HARPNUM to NOAANUM
    return a np array
    """
    noaa = hnmap[hnmap[:,0]==harp,1]
    return noaa

def n2h(noaa):
    """
    to transform the AR num from HARPNUM to NOAANUM
    return a np array
    """
    harp = hnmap[hnmap[:,1]==int(noaa),0]
    return harp

import pandas as pd
import datetime

cols=['arnum','level','t0','tm','tt','harp']

file_events = open(Path0 + 'info/flareinfo.txt')
event_list = file_events.readlines()
arnum_list = []
level_list = []
sublevel_list = []
t0_list = []
tm_list = []
tt_list = []

arharp = []

import re
for event in event_list:
    string = re.split('-|T|;|:',event)
    arnum_list.append(int(string[18][2:]))
    level_list.append(string[19][0])
    sublevel_list.append(float(string[19][1:4]))
    arharp.append(n2h(string[18][2:]))
    t0_list.append(datetime.datetime.strptime(string[0]+string[1]+string[2]+string[3]+string[4],'%Y%m%d%H%M'))
    tm_list.append(datetime.datetime.strptime(string[6]+string[7]+string[8]+string[9]+string[10],'%Y%m%d%H%M'))
    tt_list.append(datetime.datetime.strptime(string[12]+string[13]+string[14]+string[15]+string[16],'%Y%m%d%H%M'))

data = pd.DataFrame({'arnum': arnum_list, 'level': level_list, 'sublevel':sublevel_list, 't0': t0_list, 'tm': tm_list, 'tt': tt_list, 'harp':arharp})
level_list = data['level'].values.tolist()  
dict(zip(*np.unique(level_list, return_counts=True)))    
print(data['level'].value_counts())  # 查看英语列各元素出现的次数
index_M = data[data['level']=='M'].index
index_X = data[data['level']=='X'].index
index_M = np.concatenate((index_M,index_X),axis=0)
data_M = data.loc[index_M]
data_M = data_M.reset_index(drop=True)





# =============================================================================
# 输出M5级别以上的耀斑列表
# =============================================================================
# arnum_train = []
ar_num = []
ar_lv = []
ar_sublv = []

t0_list = []
tm_list = []
tt_list = []
# 将大于M5的耀斑挑出来
for ii in range(len(data)):
    if data['level'][ii] == 'M':
        if data['sublevel'][ii] >= 5:
            ar_num.append(data['arnum'][ii])
            ar_lv.append(data['level'][ii])
            ar_sublv.append(data['sublevel'][ii])
            t0_list.append(data['t0'][ii])
            tm_list.append(data['tm'][ii])
            tt_list.append(data['tt'][ii])
    elif data['level'][ii] == 'X':
        ar_num.append(data['arnum'][ii])
        ar_lv.append(data['level'][ii])
        ar_sublv.append(data['sublevel'][ii])
        t0_list.append(data['t0'][ii])
        tm_list.append(data['tm'][ii])
        tt_list.append(data['tt'][ii])
        

data_M5 = pd.DataFrame({'arnum':ar_num,'level':ar_lv,'sublevel':ar_sublv,'t0':t0_list,'tm':tm_list,'tt':tt_list})    
dict(zip(*np.unique(ar_num, return_counts=True)))    
print(data_M5['level'].value_counts())  # 查看英语列各元素出现的次数



# =============================================================================
# 输出C7级别以上的耀斑列表
# =============================================================================
# arnum_train = []
ar_num = []
ar_lv = []
ar_sublv = []

t0_list = []
tm_list = []
tt_list = []
# 将大于M5的耀斑挑出来
for ii in range(len(data)):
    if data['level'][ii] == 'C':
        if data['sublevel'][ii] >= 7:
            ar_num.append(data['arnum'][ii])
            ar_lv.append(data['level'][ii])
            ar_sublv.append(data['sublevel'][ii])
            t0_list.append(data['t0'][ii])
            tm_list.append(data['tm'][ii])
            tt_list.append(data['tt'][ii])
    elif data['level'][ii] == 'M':
        ar_num.append(data['arnum'][ii])
        ar_lv.append(data['level'][ii])
        ar_sublv.append(data['sublevel'][ii])
        t0_list.append(data['t0'][ii])
        tm_list.append(data['tm'][ii])
        tt_list.append(data['tt'][ii])
    elif data['level'][ii] == 'X':
        ar_num.append(data['arnum'][ii])
        ar_lv.append(data['level'][ii])
        ar_sublv.append(data['sublevel'][ii])
        t0_list.append(data['t0'][ii])
        tm_list.append(data['tm'][ii])
        tt_list.append(data['tt'][ii])
        

data_C7 = pd.DataFrame({'arnum':ar_num,'level':ar_lv,'sublevel':ar_sublv,'t0':t0_list,'tm':tm_list,'tt':tt_list})    
dict(zip(*np.unique(ar_num, return_counts=True)))    
print(data_C7['level'].value_counts())  # 查看英语列各元素出现的次数












# 加载所有训练集和测试集的时序数据
import numpy as np
dataPath = 'H:/dataset/pil/' + 'dataset_2class_now_sq/'
adv = 24
para_train = np.load(dataPath + str(adv)+'/para_train.npy')
label_train = np.load(dataPath + str(adv)+'/label_train.npy')
para_test = np.load(dataPath + str(adv)+'/para_test.npy')
label_test = np.load(dataPath + str(adv)+'/label_test.npy')
arnum_test = np.load(dataPath + str(adv)+'/arnum_test.npy')
arnum_train = np.load(dataPath + str(adv)+'/arnum_train.npy')
countf_train = np.load(dataPath + str(adv)+'/countf_train.npy')
countf_test = np.load(dataPath + str(adv)+'/countf_train.npy')

# =============================================================================
# 分别从训练集和测试集当中，选出大耀斑活动区，即C7及以上耀斑的活动区
# =============================================================================

# 训练集中的大耀斑活动区
arnum_train_noaa = [h2n(x) for x in arnum_train]
arnum_train_noaa = np.array(arnum_train_noaa).reshape((len(arnum_train_noaa),))

list_arnum_train_noaa = list(set(arnum_train_noaa))
arnum_train_noaa = np.array(list_arnum_train_noaa)
arnum_train_noaa.sort() 

arnum_train_chosen = []
for item in arnum_train_noaa:
    if item in list(data_C7['arnum']):
        arnum_train_chosen.append(item)


# 测试集中的大耀斑活动区
arnum_test_noaa = [h2n(x) for x in arnum_test]
arnum_test_noaa = np.array(arnum_test_noaa).reshape((len(arnum_test_noaa),))

list_arnum_test_noaa = list(set(arnum_test_noaa))#重新创建一个变量，接收返回值。使用list方法中的set函数
arnum_test_noaa = np.array(list_arnum_test_noaa)
arnum_test_noaa.sort() 

arnum_test_chosen = []
for item in arnum_test_noaa:
    if item in list(data_C7['arnum']):
        arnum_test_chosen.append(item)
        
        
print("在训练集中共选出 "+str(len(arnum_train_chosen))+" 个活动区")
print("在测试集中共选出 "+str(len(arnum_test_chosen))+" 个活动区")


# 数据处理必要函数
from scipy.interpolate import make_interp_spline

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
def create_dataset(xset,yset, look_back):
    '''
    对数据进行处理，针对数据量较少的1事件
    '''
    dataX, dataY = [], []
    for i in range(len(xset)-look_back+1):
        a = xset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(yset[i + look_back-1])
    Train_X = np.array(dataX)
    Train_Y = np.array(dataY)

    return Train_X, Train_Y
def stp2t(arr_tstamp):
    arr_time = []
    for item in arr_tstamp:
        arr_time.append(datetime.datetime.fromtimestamp(item))
    return np.array(arr_time)

# =============================================================================
# 将画图封装成函数
# =============================================================================


arnum_noaa = arnum_train_chosen[2]
arnum_noaa = 11158


import scipy.io as spio
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter
from tensorflow.keras.models import load_model
import joblib

def t_info(arnum_noaa):
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    
    
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    
    t_s = time_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    
    
    t_start_set = datetime.datetime.fromtimestamp(t_resamp2[0])
    t_end_set = datetime.datetime.fromtimestamp(t_resamp2[-1])
    
    print('--NOAA('+str(arnum_noaa)+')数据的时间范围--\nfrom: '+str(t_start_set)+'\n  to: '+str(t_end_set))
    
    

# t_start_new = datetime.datetime(2016,8,7)
# t_end_new = datetime.datetime(2016,8,17)



# tt = datetime.datetime.strptime('20181204_181200','%Y%m%d_%H%M%S')
# t_start_set.strftime('%Y/%m/%dT%H:%M')
# datetime.datetime.fromtimestamp(item)
fold_selected = 5
trange_sqs = [6,12,18,24,30,36,42,48]
# step2: 定义所有需要比较的模型的名称
model_names = [f'model_cnn-lstm-at_now_6h_fold{fold_selected}.h5',
               f'model_cnn-lstm-at_now_12h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_18h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_24h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_30h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_36h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_42h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_48h_fold{fold_selected}.h5']

import os
from keras import backend as K
from keras import models
from keras.models import load_model
import tensorflow as tf

import scipy.io as spio
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter
from tensorflow.keras.models import load_model
import joblib

def t_prob_plot(arnum_noaa,trange_sq=24,fold_selected=2,
                save_path='./',
                save=False,imshow=True,time_range=0):

    adv=24

    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+f"now_{trange_sq}h"+'_sq/'
    
    model_name = f'model_cnn-lstm-at_now_{trange_sq}h_fold{fold_selected}.h5'
    
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    t0s =  data['t0'][index_data_C]
    t0s_C = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    t0s =  data['t0'][index_data_M]
    t0s_M = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    keys = list(s.keys())
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    # 选择数据中有用的列
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    
    t_s = time_selected
    p_s = paras_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    p_c = p_resamp1[int(n_del)::]
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
    
    t_plot = t_resamp2
    c_plot = 2
    p_plot = p_resamp2[:,c_plot]
    
    
    l_resamp2 = np.ones(len(p_resamp2))*1
    p_data,_ = create_dataset(p_resamp2,l_resamp2, look_back=trange_sq+1)
    
    # 导入原来的数据集的归一化
    scalarX = joblib.load(dataPath +str(adv)+'/scale_'+f'minmax_fold{fold_selected}'+'.save') 
    # 导入模型
    # file_pre = dataPath  + str(adv) + "/check_10_0.2132_cnn-lstm-at-ppt.hdf5"
    # file_pre = 'H:/dataset/pil/dataset_2class_now_sq/24/model_cnn-lstm-at-now-new0.h5'
    file_pre_all = os.listdir(dataPath  + str(adv))
    file_pres = []
    for each_filename in file_pre_all:
        if 'check_100' and f'fold{fold_selected}' in each_filename:
            file_pres.append(each_filename)
    
    model_path = dataPath + str(24)+'/'+file_pres[0]
    
    m = load_model(model_path)
    
    
    p_2d = p_data.reshape((p_data.shape[0]*(trange_sq+1),20))
    p_test = scalarX.transform(p_2d)
    
    p_3d = p_test.reshape((p_data.shape[0],trange_sq+1,20))
    # 只选取后面参数，不选PIL_mask的长宽
    p_3d = p_3d[:,:,2::]
    pred = m.predict([p_3d])
    # pred_label = pred.argmax(1)
    
    prob_1 = pred[:,1]
    pred_stp = np.array(t_resamp2[trange_sq::]).reshape((len(t_resamp2[trange_sq::]),))
    pred_t = stp2t(pred_stp)
    
    
    fig=plt.figure(figsize=(8,5),dpi=300)
    ax1=fig.add_subplot(111)
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    font = {'family':'Times New Roman','weight':'normal','size':15}
    ax1.xaxis.set_major_formatter(DateFormatter('%d'))  #\n%H:%M')) # '%Y%m%d%H%M'
    ax1.set_xlabel('day',fontsize=20,fontproperties="Times New Roman")
    ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$',fontsize=20,fontproperties="Times New Roman")
    
    t_max = max(stp2t(t_plot))
    t_min = min(stp2t(t_plot))
    plt.title('NOAA'+str(arnum_noaa)+'\n'+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
              fontproperties="Times New Roman",fontsize=20)
    h1 = ax1.plot(stp2t(t_plot),p_plot,'royalblue',label='para',linestyle='--',marker='.',markersize=8)
    ixs = 0
    for each_t0 in list(t0s_C):
        h3 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],color='gold',linestyle='-.',label='$t_0$ of C-class flare')
        # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
        ixs += 1
    for each_t0 in list(t0s_M):
        h2 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='$t_0$ of M/X-class flare')
        # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
        ixs += 1
    plt.xticks(fontsize=18,fontproperties="Times New Roman",rotation=0)
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    ax2 = ax1.twinx() # 包含另一个y轴的坐标轴对象
    h4 = ax2.plot(pred_t,prob_1,'g',label='pred')
    ax2.set_ylabel('probability',fontsize=20,fontproperties="Times New Roman")
    # ax2.set_xlim(0,1)
    ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax2.grid(linestyle="--",axis='y')
    if len(t0s_M) > 0:
        if len(t0s_C) > 0:
            keys = h1 + h4 + h2 + h3
        else:
            keys = h1 + h4 + h2
    elif len(t0s_C) > 0:
        keys = h1 + h4 + h3
    else:
        keys = h1 + h4

    labs = [l.get_label() for l in keys]
    legend = ax1.legend(keys,labs,prop=font,loc = 'upper left',bbox_to_anchor=(0.01, 0.99))
    handles = legend.legendHandles
    for i in range(len(handles)):
        # label = handles[i].get_label()
        handles[i].set_alpha(0.5)

    # ax1.set_yscale('log')
    
    time_range = [datetime.datetime(2013,8,6,7),
                  datetime.datetime(2013,8,19,0)]
    
    t_plot_s = []
    y_plot_s = []
    for ix_item in range(len(t_plot)):
        if stp2t(t_plot)[ix_item] >=time_range[0] and stp2t(t_plot)[ix_item] <=time_range[1]:
            t_plot_s.append(stp2t(t_plot)[ix_item])
            y_plot_s.append(p_plot[ix_item])
        
    max_set = max(y_plot_s)
    min_set = 0
    ax1.set_ylim(min_set,max_set*1.05)
    
    ax1.set_ylim(0,100)

    ax1.set_xlim(time_range[0]-(time_range[1]-time_range[0])*0.05,time_range[1]+(time_range[1]-time_range[0])*0.05)
    
    
    if time_range == 0:
        ax2.set_xlim(t_min-(t_max-t_min)*0.05,t_max+(t_max-t_min)*0.05)
    else:
        ax2.set_xlim(time_range[0],time_range[1])
    
    plt.xticks(fontsize=16,fontproperties="Times New Roman")
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    plt.tight_layout()
    
    
    if save: 
        # plt.savefig(save_path+str(arnum_noaa)+'.jpg',bbox_inches='tight')
        plt.savefig("E:/磁参时序/forshow/pred_"+str(trange_sq)+".pdf", format='pdf')
    if not imshow:
        plt.clf()
        plt.close('all')
    else:
        plt.show()




t_prob_plot(12257,trange_sq=24,fold_selected=5,save=False,imshow=True)





fold_selected = 5
arnum_test = np.load('H:/dataset/pil/info/' + f'ar_test{fold_selected}.npy')

sss = []
for arr in arnum_test:
    if h2n(arr)[0] in list(data_M['arnum']):
        sss.append(arr)
        
ar_noaa = h2n(sss[7])[0]

t_prob_plot(ar_noaa,trange_sq=24,fold_selected=fold_selected,save=False,imshow=True)

trange_sqs = [6,12,18,24,30,36,42,48]

for ttt in range(len(trange_sqs)):
    trange_sq = trange_sqs[ttt]
    t_prob_plot(ar_noaa,trange_sq=trange_sq,fold_selected=fold_selected,
                save=True,imshow=True)







def t_prob_plot1(arnum_noaa,trange_sq=24,fold_selected=2,
                save_path='./',
                save=False,imshow=True,time_range=0):

    adv=24

    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+"now_24h"+'_sq/'
    
    model_name = 'check_10_0.2132_cnn-lstm-at-ppt.hdf5'
    
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    t0s =  data['t0'][index_data_C]
    t0s_C = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    t0s =  data['t0'][index_data_M]
    t0s_M = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    keys = list(s.keys())
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    # 选择数据中有用的列
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    
    t_s = time_selected
    p_s = paras_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    p_c = p_resamp1[int(n_del)::]
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
    
    t_plot = t_resamp2
    c_plot = 2
    p_plot = p_resamp2[:,c_plot]
    
    
    l_resamp2 = np.ones(len(p_resamp2))*1
    p_data,_ = create_dataset(p_resamp2,l_resamp2, look_back=trange_sq+1)
    
    # 导入原来的数据集的归一化
    scalarX = joblib.load(dataPath +str(adv)+'/'+f'scale_minmax.save') 
    # 导入模型
    file_pre = dataPath  + str(adv) + "/check_10_0.2132_cnn-lstm-at-ppt.hdf5"
    # file_pre = 'H:/dataset/pil/dataset_2class_now_sq/24/model_cnn-lstm-at-now-new0.h5'
    file_pre_all = os.listdir(dataPath  + str(adv))
    file_pres = []
    for each_filename in file_pre_all:
        if 'check_10' and f'fold{fold_selected}' in each_filename:
            file_pres.append(each_filename)
    
    model_path = dataPath + str(24)+'/'+file_pres[0]
    
    m = load_model(file_pre)
    
    
    p_2d = p_data.reshape((p_data.shape[0]*(trange_sq+1),20))
    p_test = scalarX.transform(p_2d)
    
    p_3d = p_test.reshape((p_data.shape[0],trange_sq+1,20))
    # 只选取后面参数，不选PIL_mask的长宽
    p_3d = p_3d[:,:,2::]
    pred = m.predict([p_3d])
    # pred_label = pred.argmax(1)
    
    prob_1 = pred[:,1]
    pred_stp = np.array(t_resamp2[trange_sq::]).reshape((len(t_resamp2[trange_sq::]),))
    pred_t = stp2t(pred_stp)
    
    
    fig=plt.figure(figsize=(8,5),dpi=300)
    ax1=fig.add_subplot(111)
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    font = {'family':'Times New Roman','weight':'normal','size':15}
    ax1.xaxis.set_major_formatter(DateFormatter('%d'))  #\n%H:%M')) # '%Y%m%d%H%M'
    ax1.set_xlabel('day',fontsize=20,fontproperties="Times New Roman")
    ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$',fontsize=20,fontproperties="Times New Roman")
    
    t_max = max(stp2t(t_plot))
    t_min = min(stp2t(t_plot))
    plt.title('NOAA'+str(arnum_noaa)+'\n'+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
              fontproperties="Times New Roman",fontsize=20)
    h1 = ax1.plot(stp2t(t_plot),p_plot,'royalblue',label='para',linestyle='--',marker='.',markersize=8)
    ixs = 0
    for each_t0 in list(t0s_C):
        h3 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],color='gold',linestyle='-.',label='$t_0$ of C-class flare')
        # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
        ixs += 1
    for each_t0 in list(t0s_M):
        h2 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='$t_0$ of M/X-class flare')
        # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
        ixs += 1
    plt.xticks(fontsize=18,fontproperties="Times New Roman",rotation=0)
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    ax2 = ax1.twinx() # 包含另一个y轴的坐标轴对象
    h4 = ax2.plot(pred_t,prob_1,'g',label='pred')
    ax2.set_ylabel('probability',fontsize=20,fontproperties="Times New Roman")
    # ax2.set_xlim(0,1)
    ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax2.grid(linestyle="--",axis='y')
    if len(t0s_M) > 0:
        if len(t0s_C) > 0:
            keys = h1 + h4 + h2 + h3
        else:
            keys = h1 + h4 + h2
    elif len(t0s_C) > 0:
        keys = h1 + h4 + h3
    else:
        keys = h1 + h4

    labs = [l.get_label() for l in keys]
    legend = ax1.legend(keys,labs,prop=font,loc = 'upper left',bbox_to_anchor=(0.01, 0.99))
    handles = legend.legendHandles
    for i in range(len(handles)):
        # label = handles[i].get_label()
        handles[i].set_alpha(0.5)

    # ax1.set_yscale('log')
    
    # time_range = [datetime.datetime(2013,8,6,7),
    #               datetime.datetime(2013,8,19,0)]
    
    # t_plot_s = []
    # y_plot_s = []
    # for ix_item in range(len(t_plot)):
    #     if stp2t(t_plot)[ix_item] >=time_range[0] and stp2t(t_plot)[ix_item] <=time_range[1]:
    #         t_plot_s.append(stp2t(t_plot)[ix_item])
    #         y_plot_s.append(p_plot[ix_item])
        
    # max_set = max(y_plot_s)
    # min_set = 0
    # ax1.set_ylim(min_set,max_set*1.05)
    
    # ax1.set_ylim(0,100)

    # ax1.set_xlim(time_range[0]-(time_range[1]-time_range[0])*0.05,time_range[1]+(time_range[1]-time_range[0])*0.05)
    
    
    if time_range == 0:
        ax2.set_xlim(t_min,t_max)
    else:
        ax2.set_xlim(time_range[0],time_range[1])
    
    plt.xticks(fontsize=16,fontproperties="Times New Roman")
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    plt.tight_layout()
    
    
    if save: 
        # plt.savefig(save_path+str(arnum_noaa)+'.jpg',bbox_inches='tight')
        plt.savefig(f"E:/磁参时序/forshow/show_{arnum_noaa}"+".pdf", format='pdf')
    if not imshow:
        plt.clf()
        plt.close('all')
    else:
        plt.show()


t_prob_plot1(12422,fold_selected=fold_selected,save=True,imshow=True)

t_prob_plot1(12257,fold_selected=fold_selected,save=True,imshow=True)


t_prob_plot1(11158,fold_selected=fold_selected,save=False,imshow=True)






def t_prob_plot2(arnum_noaa,trange_sq=24,fold_selected=2,
                save_path='./',
                save=False,imshow=True,time_range=0):

    adv=24

    dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+"now_24h"+'_sq/'
    
    model_name = 'check_10_0.2132_cnn-lstm-at-ppt.hdf5'
    
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    t0s =  data['t0'][index_data_C]
    t0s_C = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    t0s =  data['t0'][index_data_M]
    t0s_M = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    keys = list(s.keys())
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    # 选择数据中有用的列
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    
    t_s = time_selected
    p_s = paras_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    p_c = p_resamp1[int(n_del)::]
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
    
    t_plot = t_resamp2
    c_plot = 2
    p_plot = p_resamp2[:,c_plot]
    
    
    l_resamp2 = np.ones(len(p_resamp2))*1
    p_data,_ = create_dataset(p_resamp2,l_resamp2, look_back=trange_sq+1)
    
    # 导入原来的数据集的归一化
    scalarX = joblib.load(dataPath +str(adv)+'/'+f'scale_minmax.save') 
    # 导入模型
    file_pre = dataPath  + str(adv) + "/check_10_0.2132_cnn-lstm-at-ppt.hdf5"
    # file_pre = 'H:/dataset/pil/dataset_2class_now_sq/24/model_cnn-lstm-at-now-new0.h5'
    file_pre_all = os.listdir(dataPath  + str(adv))
    file_pres = []
    for each_filename in file_pre_all:
        if 'check_10' and f'fold{fold_selected}' in each_filename:
            file_pres.append(each_filename)
    
    model_path = dataPath + str(24)+'/'+file_pres[0]
    
    m = load_model(file_pre)
    
    
    p_2d = p_data.reshape((p_data.shape[0]*(trange_sq+1),20))
    p_test = scalarX.transform(p_2d)
    
    p_3d = p_test.reshape((p_data.shape[0],trange_sq+1,20))
    # 只选取后面参数，不选PIL_mask的长宽
    p_3d = p_3d[:,:,2::]
    pred = m.predict([p_3d])
    # pred_label = pred.argmax(1)
    
    prob_1 = pred[:,1]
    pred_stp = np.array(t_resamp2[trange_sq::]).reshape((len(t_resamp2[trange_sq::]),))
    pred_t = stp2t(pred_stp)
    
        
        # 假设其他必要的导入和数据准备工作已经完成
    
    t_total = len(pred_t)
    
    for current_point in range(1, t_total + 1):
        fig = plt.figure(figsize=(8, 5), dpi=300)
        ax1 = fig.add_subplot(111)
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.sans-serif'] = "Times New Roman"
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
        ax1.xaxis.set_major_formatter(DateFormatter('%d'))
        ax1.set_xlabel('day', fontsize=20, fontproperties="Times New Roman")
        ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$', fontsize=20, fontproperties="Times New Roman")
        
        t_max = max(stp2t(t_plot))
        t_min = min(stp2t(t_plot))
        plt.title('NOAA' + str(arnum_noaa) + '\n' + t_min.strftime('%Y/%m/%d') + '-' + t_max.strftime('%Y/%m/%d'),
                  fontproperties="Times New Roman", fontsize=20)
        h1 = ax1.plot(stp2t(t_plot), p_plot, 'royalblue', label='para', linestyle='--', marker='.', markersize=8)
        
        ixs = 0
        for each_t0 in list(t0s_C):
            h3 = ax1.plot([each_t0, each_t0], [min(p_plot), max(p_plot)], color='gold', linestyle='-.', label='$t_0$ of C-class flare')
            ixs += 1
        for each_t0 in list(t0s_M):
            h2 = ax1.plot([each_t0, each_t0], [min(p_plot), max(p_plot)], 'r-.', label='$t_0$ of M/X-class flare')
            ixs += 1
        
        plt.xticks(fontsize=18, fontproperties="Times New Roman", rotation=0)
        plt.yticks(fontsize=20, fontproperties="Times New Roman")
        
        ax2 = ax1.twinx()  # 包含另一个y轴的坐标轴对象
        h4 = ax2.plot(pred_t[:current_point], prob_1[:current_point], 'g', label='pred')
    
        # 标记当前端点
        current_value = prob_1[current_point - 1]
        color = 'darkgreen' if current_value < 0.5 else 'red'
        ax2.plot(pred_t[current_point - 1], current_value, marker='*', color=color, markersize=10)
        
        ax2.set_ylabel('probability', fontsize=20, fontproperties="Times New Roman")
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.grid(linestyle="--", axis='y')
        
        if len(t0s_M) > 0:
            if len(t0s_C) > 0:
                keys = h1 + h4 + h2 + h3
            else:
                keys = h1 + h4 + h2
        elif len(t0s_C) > 0:
            keys = h1 + h4 + h3
        else:
            keys = h1 + h4
    
        labs = [l.get_label() for l in keys]
        legend = ax1.legend(keys, labs, prop=font, loc='upper left', bbox_to_anchor=(0.01, 0.99))
        handles = legend.legendHandles
        for i in range(len(handles)):
            handles[i].set_alpha(0.5)
        
        if time_range == 0:
            ax2.set_xlim(t_min, t_max)
        else:
            ax2.set_xlim(time_range[0], time_range[1])
        
        plt.xticks(fontsize=16, fontproperties="Times New Roman")
        plt.yticks(fontsize=20, fontproperties="Times New Roman")
        plt.tight_layout()
        
        plt.savefig(f"D:/table/forshow/tu{current_point}"+".png", format='png')
        plt.clf()
        plt.close(fig)
    
    
    
    # fig=plt.figure(figsize=(8,5),dpi=300)
    # ax1=fig.add_subplot(111)
    # plt.rcParams['mathtext.default'] = 'regular'
    # plt.rcParams['font.sans-serif'] = "Times New Roman"
    # font = {'family':'Times New Roman','weight':'normal','size':15}
    # ax1.xaxis.set_major_formatter(DateFormatter('%d'))  #\n%H:%M')) # '%Y%m%d%H%M'
    # ax1.set_xlabel('day',fontsize=20,fontproperties="Times New Roman")
    # ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$',fontsize=20,fontproperties="Times New Roman")
    
    # t_max = max(stp2t(t_plot))
    # t_min = min(stp2t(t_plot))
    # plt.title('NOAA'+str(arnum_noaa)+'\n'+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
    #           fontproperties="Times New Roman",fontsize=20)
    # h1 = ax1.plot(stp2t(t_plot),p_plot,'royalblue',label='para',linestyle='--',marker='.',markersize=8)
    # ixs = 0
    # for each_t0 in list(t0s_C):
    #     h3 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],color='gold',linestyle='-.',label='$t_0$ of C-class flare')
    #     # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
    #     ixs += 1
    # for each_t0 in list(t0s_M):
    #     h2 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='$t_0$ of M/X-class flare')
    #     # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
    #     ixs += 1
    # plt.xticks(fontsize=18,fontproperties="Times New Roman",rotation=0)
    # plt.yticks(fontsize=20,fontproperties="Times New Roman")
    # ax2 = ax1.twinx() # 包含另一个y轴的坐标轴对象
    # h4 = ax2.plot(pred_t,prob_1,'g',label='pred')
    # ax2.set_ylabel('probability',fontsize=20,fontproperties="Times New Roman")
    # # ax2.set_xlim(0,1)
    # ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    # ax2.grid(linestyle="--",axis='y')
    # if len(t0s_M) > 0:
    #     if len(t0s_C) > 0:
    #         keys = h1 + h4 + h2 + h3
    #     else:
    #         keys = h1 + h4 + h2
    # elif len(t0s_C) > 0:
    #     keys = h1 + h4 + h3
    # else:
    #     keys = h1 + h4

    # labs = [l.get_label() for l in keys]
    # legend = ax1.legend(keys,labs,prop=font,loc = 'upper left',bbox_to_anchor=(0.01, 0.99))
    # handles = legend.legendHandles
    # for i in range(len(handles)):
    #     # label = handles[i].get_label()
    #     handles[i].set_alpha(0.5)

    # # ax1.set_yscale('log')
    
    # # time_range = [datetime.datetime(2013,8,6,7),
    # #               datetime.datetime(2013,8,19,0)]
    
    # # t_plot_s = []
    # # y_plot_s = []
    # # for ix_item in range(len(t_plot)):
    # #     if stp2t(t_plot)[ix_item] >=time_range[0] and stp2t(t_plot)[ix_item] <=time_range[1]:
    # #         t_plot_s.append(stp2t(t_plot)[ix_item])
    # #         y_plot_s.append(p_plot[ix_item])
        
    # # max_set = max(y_plot_s)
    # # min_set = 0
    # # ax1.set_ylim(min_set,max_set*1.05)
    
    # # ax1.set_ylim(0,100)

    # # ax1.set_xlim(time_range[0]-(time_range[1]-time_range[0])*0.05,time_range[1]+(time_range[1]-time_range[0])*0.05)
    
    
    # if time_range == 0:
    #     ax2.set_xlim(t_min,t_max)
    # else:
    #     ax2.set_xlim(time_range[0],time_range[1])
    
    # plt.xticks(fontsize=16,fontproperties="Times New Roman")
    # plt.yticks(fontsize=20,fontproperties="Times New Roman")
    # plt.tight_layout()
    
    
    # if save: 
    #     # plt.savefig(save_path+str(arnum_noaa)+'.jpg',bbox_inches='tight')
    #     plt.savefig(f"D:/table/forshow/tu{current_point}"+".pdf", format='pdf')
    # if not imshow:
    #     plt.clf()
    #     plt.close('all')
    # else:
    #     plt.show()




t_prob_plot2(11158,fold_selected=fold_selected,save=False,imshow=True)



















for ttt in len(trange_sqs):
    trange_sq = trange_sqs[ttt]
    t_prob_plot(ar_noaa,trange_sq=trange_sq,fold_selected=fold_selected,save=False,imshow=True)

    





time_ranges = [datetime.datetime(2012,9,6,0),
              datetime.datetime(2012,9,11,0)]

t_prob_plot(ar_noaa,trange_sq=24,save=False,imshow=True,time_range=time_ranges)    


trange_sqs = [6,12,18,24,30,36,42,48]

for ttt in range(len(trange_sqs)):
    trange_sq = trange_sqs[ttt]
    t_prob_plot(ar_noaa,trange_sq=trange_sq,fold_selected=fold_selected,save=False,imshow=True)

    
    





# 画出单一活动区的预测图
arnum_noaa = arnum_train_chosen[4]
print('所选活动区的NOAA为:\n'+str(arnum_noaa))
t_info(arnum_noaa)
t_prob_plot(arnum_noaa,save=False,imshow=True)
t_prob_plot(arnum_noaa,trange_sq=6,save=False,imshow=True)

time_range = [datetime.datetime(2016,8,7,0,12),
              datetime.datetime(2016,8,11,0,0)]
t_prob_plot(arnum_noaa,save=False,imshow=True,time_range=time_range)        




fold_selected = 2

# # rootPath = '/data/wangjj/dataset/para_PIL/'
# dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
arnum_test = np.load('H:/dataset/pil/info/' + f'ar_test{fold_selected}.npy')


sss = []
for arr in arnum_test:
    if arr in data_C7['arnum']:
        sss.append(arr)



ar_noaa = sss[2]




from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import layers
from keras import models
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# 不同的时间窗长度将采用不同的模型
# 再画出不同模型下的预测曲线图，即可比较不同长度的时间窗下的预报效果的细节
# step1: 设置sequence的时间长度
trange_sqs = [6,12,18,24,30,36,42,48]
# step2: 定义所有需要比较的模型的名称
model_names = [f'model_cnn-lstm-at_now_6h_fold{fold_selected}.h5',
               f'model_cnn-lstm-at_now_12h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_18h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_24h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_30h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_36h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_42h_fold{fold_selected}.h5',
                f'model_cnn-lstm-at_now_48h_fold{fold_selected}.h5']

def mult_prob_plot(arnum_noaa,trange_sqs=trange_sqs,
                   model_names=model_names,
                save_path='./',
                   save=False,imshow=True,time_range=0):


    
    
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    t0s =  data['t0'][index_data_C]
    t0s_C = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    t0s =  data['t0'][index_data_M]
    t0s_M = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    keys = list(s.keys())
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    # 选择数据中有用的列
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    
    t_s = time_selected
    p_s = paras_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    p_c = p_resamp1[int(n_del)::]
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
    
    t_plot = t_resamp2
    c_plot = 2
    p_plot = p_resamp2[:,c_plot]
    
    

    
    
    index_data = data[data['arnum']==arnum_noaa].index
    index_data_C = []
    index_data_M = []
    for i in range(len(index_data)): 
        if data['level'][index_data[i]] == 'C':
            index_data_C.append(index_data[i])
        elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
            index_data_M.append(index_data[i])
    
    t0s =  data['t0'][index_data_C]
    t0s_C = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    t0s =  data['t0'][index_data_M]
    t0s_M = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    
    datatype = 'now' 
    # adv_t = [24] 
    Path0 = 'H:/dataset/pil/'
    filePath = Path0 + 'parastru/'
    
    arnum_harp = n2h(arnum_noaa)[0]
    index_ar = list(hnmap[:,0]).index(arnum_harp)
    
    name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
    file = filePath + name
    s = spio.readsav(file, python_dict=True, verbose=False)
    keys = list(s.keys())
    record_list = list(s[datatype+'_parastru0'])
    time_list = []
    para_list = []
    # label_list = []
    for rec_ix in range(len(tstamp_list[index_ar])):
        # noaa = h2n(arnum_harp)
        ts = tstamp_list[index_ar][rec_ix]
        item_para = list(record_list[rec_ix])[1::]
        para_list.append(np.array(item_para))
        time_list.extend([ts])
    
    # 读取出了所有参数和时间戳信息
    paras = np.array(para_list)
    times = np.array(time_list)
    
    # 保留不是nan的数据列
    # 并且将补零的无用数据全删除
    # 选择数据中有用的列
    column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
    df_paras = pd.DataFrame(paras)
    paras_nonan = df_paras[column_selected]
    paras_selected = paras_nonan.dropna(axis=0, how='any')
    index_selected = paras_selected.index
    time_selected = times[index_selected]
    
    t_s = time_selected
    p_s = paras_selected
    t_len = max(t_s)-min(t_s)
    t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
    p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)
    # 剔除1小时的余数
    n_del = (t_len%3600)/720
    p_c = p_resamp1[int(n_del)::]
    t_c = t_resamp1[int(n_del)::]
    # 1小时采样分辨率 
    t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
    p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)
    
    t_plot = t_resamp2
    c_plot = 2
    p_plot = p_resamp2[:,c_plot]
    
    
    for ix in range(len(trange_sqs)):
        
        trange_sq = trange_sqs[ix]
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+f"now_{trange_sq}h"+'_sq/'
        
        l_resamp2 = np.ones(len(p_resamp2))*1
        p_data,_ = create_dataset(p_resamp2,l_resamp2, look_back=trange_sq+1)
        
        # 导入原来的数据集的归一化
        scalarX = joblib.load(dataPath +str(adv)+'/scale_'+f'minmax_fold{fold_selected}'+'.save') 
        # 导入模型
        # file_pre = dataPath  + str(adv) + "/check_10_0.2132_cnn-lstm-at-ppt.hdf5"
        # file_pre = 'H:/dataset/pil/dataset_2class_now_sq/24/model_cnn-lstm-at-now-new0.h5'
        fil_pre = dataPath + str(adv) + '/' + model_names[ix]
        m = load_model(fil_pre)
        
        
        p_2d = p_data.reshape((p_data.shape[0]*(trange_sq+1),20))
        p_test = scalarX.transform(p_2d)
        
        p_3d = p_test.reshape((p_data.shape[0],trange_sq+1,20))
        # 只选取后面参数，不选PIL_mask的长宽
        p_3d = p_3d[:,:,2::]
        pred = m.predict([p_3d])
        # pred_label = pred.argmax(1)
        
        prob_1 = pred[:,1]
        pred_stp = np.array(t_resamp2[trange_sq::]).reshape((len(t_resamp2[trange_sq::]),))
        pred_t = stp2t(pred_stp)
        
        # 赋值语句
        code = f'''
pred_t_{trange_sq} = pred_t
prob_1_{trange_sq} = prob_1
        '''
        exec(code)
    
    
    fig=plt.figure(figsize=(8,5),dpi=300)
    ax1=fig.add_subplot(111)
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    font = {'family':'Times New Roman','weight':'normal','size':15}
    ax1.xaxis.set_major_formatter(DateFormatter('%d')) # '%Y%m%d%H%M'
    ax1.set_xlabel('day',fontsize=20,fontproperties="Times New Roman")
    ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$',fontsize=20,fontproperties="Times New Roman")
    
    t_max = max(stp2t(t_plot))
    t_min = min(stp2t(t_plot))
    plt.title('NOAA'+str(arnum_noaa)+'\n'+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
              fontproperties="Times New Roman",fontsize=20)
    h1 = ax1.plot(stp2t(t_plot),p_plot,'royalblue',label='para',linestyle='--',marker='.',markersize=8)
    ixs = 0
    for each_t0 in list(t0s_C):
        h3 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],color='gold',linestyle='-.',label='$t_0$ of C-class flare')
        # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
        ixs += 1
    for each_t0 in list(t0s_M):
        h2 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='$t_0$ of M/X-class flare')
        # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
        ixs += 1
    plt.xticks(fontsize=20,fontproperties="Times New Roman")
    plt.yticks(fontsize=20,fontproperties="Times New Roman")
    ax2 = ax1.twinx() # 包含另一个y轴的坐标轴对象
    colors = ['g','blue','black','darkviolet','lightpink']
    hs = []
    for ix in range(len(trange_sqs)):
        trange_sq = trange_sqs[ix]
        code = '''
h'''+str(4+ix)+f''' = ax2.plot(pred_t_{trange_sq},prob_1_{trange_sq},colors[ix],label=f'pred_of {trange_sq}-hour seq')
        '''
        exec(code)
        
    code_rest = '''  
ax2.set_ylabel('probability',fontsize=20,fontproperties="Times New Roman")
# ax2.set_xlim(0,1)
ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax2.grid(linestyle="--",axis='y')
if len(t0s_M) > 0:
    if len(t0s_C) > 0:
        keys = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8
    else:
        keys = h1 + h2 + h4 + h5 + h6 + h7 + h8
elif len(t0s_C) > 0:
    keys = h1 + h3 + h4 + h5 + h6 + h7 + h8 
else:
    keys = h1 + h4 + h5 + h6 + h7 + h8

labs = [l.get_label() for l in keys]
legend = ax1.legend(keys,labs,prop=font,loc = 'upper left',bbox_to_anchor=(0.01, 0.99))
legend.set_zorder(11)
handles = legend.legendHandles
for i in range(len(handles)):
    # label = handles[i].get_label()
    handles[i].set_alpha(0.5)

if time_range == 0:
    ax1.set_xlim(t_min-(t_max-t_min)*0.05,t_max+(t_max-t_min)*0.05)
else:
    ax1.set_xlim(time_range[0]-(time_range[1]-time_range[0])*0.05,time_range[1]+(time_range[1]-time_range[0])*0.05)
plt.xticks(fontsize=20,fontproperties="Times New Roman")
plt.yticks(fontsize=20,fontproperties="Times New Roman")
plt.tight_layout()
if save: 
    plt.savefig(save_path+str(arnum_noaa)+'.jpg',bbox_inches='tight')

if not imshow:
    plt.clf()
    plt.close('all')
else:
    plt.show()
    '''
    exec(code_rest)


ar_noaa = sss[0]
ar_noaa = h2n(ar_noaa)[0]
mult_prob_plot(ar_noaa,save=False,imshow=True)










mult_prob_plot(arnum_train_chosen[3],save=False,imshow=True)



from tqdm import trange
# 批量画出train和test
import warnings
warnings.filterwarnings("ignore")
for ip in trange(len(arnum_train_chosen)):
    save_path = 'E:/磁参时序/forshow/time-para-prob/train_set/'
    t_prob_plot(arnum_train_chosen[ip],save_path=save_path,save=True,imshow=False)
    
for ip in trange(len(arnum_test_chosen)):
    save_path = 'E:/磁参时序/forshow/time-para-prob/test_set/'
    t_prob_plot(arnum_test_chosen[ip],save_path=save_path,save=True,imshow=False)


# # 挑选一个arnum进行预报概率可视化
# arnum_noaa = arnum_test_chosen[1]
# arnum_noaa = arnum_train_chosen[1]



# arnum_noaa = arnum_test_chosen[1]
# arnum_noaa = arnum_train_chosen[1]






# index_data = data[data['arnum']==arnum_noaa].index
# index_data_C = []
# index_data_M = []
# for i in range(len(index_data)): 
#     if data['level'][index_data[i]] == 'C':
#         index_data_C.append(index_data[i])
#     elif data['level'][index_data[i]] == 'M' or data['level'][index_data[i]] == 'X':
#         index_data_M.append(index_data[i])
    



# t0s =  data['t0'][index_data_C]
# t0s_C = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

# t0s =  data['t0'][index_data_M]
# t0s_M = stp2t((t0s-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))




# datatype = 'now' 
# adv_t = [24] 
# Path0 = 'H:/dataset/pil/'
# filePath = Path0 + 'parastru/'
# import scipy.io as spio

# arnum_harp = n2h(arnum_noaa)[0]
# index_ar = list(hnmap[:,0]).index(arnum_harp)

# name = 'hmi.sharp_cea_720s.'+str(arnum_harp)+'.sharp_now_parastru.sav'
# file = filePath + name
# s = spio.readsav(file, python_dict=True, verbose=False)
# keys = list(s.keys())
# record_list = list(s[keys[1]])
# time_list = []
# para_list = []
# # label_list = []
# for rec_ix in range(len(tstamp_list[index_ar])):
#     noaa = h2n(arnum_harp)
#     ts = tstamp_list[index_ar][rec_ix]
#     # label = classify_M(noaa,ts,adv)
#     item_para = list(record_list[rec_ix])[1::]
    
#     para_list.append(np.array(item_para))
#     # label_list.extend([label])
#     time_list.extend([ts])

# # 读取出了所有参数和时间戳信息
# paras = np.array(para_list)
# times = np.array(time_list)
# # 保留不是nan的数据列
# # 并且将补零的无用数据全删除
# column_selected = [0,1,2,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,23,24]
# df_paras = pd.DataFrame(paras)
# paras_nonan = df_paras[column_selected]
# paras_selected = paras_nonan.dropna(axis=0, how='any')
# index_selected = paras_selected.index
# time_selected = times[index_selected]

# t_s = time_selected
# p_s = paras_selected

# t_len = max(t_s)-min(t_s)

# t_resamp1 = np.linspace(t_s[0],t_s[-1],int((t_s[-1]-t_s[0])/720+1))
# p_resamp1 = multinterp(t_s,p_s,t_resamp1,k=1)

# # 剔除1小时的余数
# n_del = (t_len%3600)/720
# p_c = p_resamp1[int(n_del)::]
# t_c = t_resamp1[int(n_del)::]

# # 1小时采样分辨率
# t_resamp2 = np.linspace(t_c[0],t_c[-1],int((t_c[-1]-t_c[0])/3600+1))
# p_resamp2 = multinterp(t_c,p_c,t_resamp2,k=1)


# import matplotlib.pyplot as plt
# from matplotlib.dates import  DateFormatter



# t_plot = t_resamp2
# c_plot = 2
# p_plot = p_resamp2[:,c_plot]

# # fig=plt.figure(dpi=300)
# # ax1=fig.add_subplot(111)
# # ax1.xaxis.set_major_formatter(DateFormatter('%d')) # '%Y%m%d%H%M'
# # t_max = max(stp2t(t_plot))
# # year_max = t_max.year
# # t_min = min(stp2t(t_plot))
# # plt.title('(NOAA'+str(arnum_noaa)+') '+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
# #           fontproperties="Times New Roman",fontsize=20)
# # h1 = plt.plot(stp2t(t_plot),p_plot,label='para')
# # for each_t0 in list(t0s):
# #     h2 = plt.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='t0 of flare')
# # keys = h1 + h2
# # labs = [l.get_label() for l in keys]
# # font = {'family':'Times New Roman','weight':'normal','size':15}
# # ax1.legend(keys,labs,prop=font)
# # ax1.set_xlim(t_min-(t_max-t_min)*0.05,t_max+(t_max-t_min)*0.05)
# # plt.xticks(fontsize=20,fontproperties="Times New Roman")
# # plt.yticks(fontsize=20,fontproperties="Times New Roman")
# # plt.show()


# l_resamp2 = np.ones(len(p_resamp2))*1
# p_data,_ = create_dataset(p_resamp2,l_resamp2, look_back=25)

# import joblib
# scalarX = joblib.load(dataPath +str(adv)+'/scale_'+'minmax'+'.save') 
# from tensorflow.keras.models import load_model
# file_pre = dataPath  + str(adv) + "/check_10_0.2132_cnn-lstm-at-ppt.hdf5"
# m = load_model(file_pre)


# p_2d = p_data.reshape((p_data.shape[0]*25,20))
# p_test = scalarX.transform(p_2d)

# p_3d = p_test.reshape((p_data.shape[0],25,20))
# # 只选取后面参数，不选PIL_mask的长宽
# p_3d = p_3d[:,:,2::]
# pred = m.predict([p_3d])
# pred_label = pred.argmax(1)

# prob_1 = pred[:,1]
# pred_stp = np.array(t_resamp2[24::]).reshape((len(t_resamp2[24::]),))
# pred_t = stp2t(pred_stp)

# save = True
# imshow = False

# fig=plt.figure(figsize=(8,5),dpi=300)
# ax1=fig.add_subplot(111)
# plt.rcParams['mathtext.default'] = 'regular'
# plt.rcParams['font.sans-serif'] = "Times New Roman"
# font = {'family':'Times New Roman','weight':'normal','size':15}
# ax1.xaxis.set_major_formatter(DateFormatter('%d')) # '%Y%m%d%H%M'
# ax1.set_xlabel('day',fontsize=20,fontproperties="Times New Roman")
# ax1.set_ylabel('TOTUSJH $(G^2m^{-1})$',fontsize=20,fontproperties="Times New Roman")

# t_max = max(stp2t(t_plot))
# year_max = t_max.year
# t_min = min(stp2t(t_plot))
# plt.title('NOAA'+str(arnum_noaa)+'\n'+t_min.strftime('%Y/%m/%d')+'-'+t_max.strftime('%Y/%m/%d'),
#           fontproperties="Times New Roman",fontsize=20)
# h1 = ax1.plot(stp2t(t_plot),p_plot,'royalblue',label='para',linestyle='--',marker='.',markersize=8)
# ixs = 0
# for each_t0 in list(t0s_M):
#     h2 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],'r-.',label='$t_0$ of M/X-class flare')
#     # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
#     ixs += 1
# for each_t0 in list(t0s_C):
#     h3 = ax1.plot([each_t0,each_t0],[min(p_plot),max(p_plot)],color='gold',linestyle='-.',label='$t_0$ of C-class flare')
#     # plt.text(each_t0-(t_max-t_min)*0.01, min(p_plot), str(ixs+1), color='r',fontdict = font,ha='right')
#     ixs += 1
# plt.xticks(fontsize=20,fontproperties="Times New Roman")
# plt.yticks(fontsize=20,fontproperties="Times New Roman")
# ax2 = ax1.twinx() # 包含另一个y轴的坐标轴对象
# h4 = ax2.plot(pred_t,prob_1,'g',label='pred')
# ax2.set_ylabel('probability',fontsize=20,fontproperties="Times New Roman")
# ax2.set_xlim(0,1)
# ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])

# keys = h1 + h2 + h3 + h4
# labs = [l.get_label() for l in keys]
# legend = ax1.legend(keys,labs,prop=font,loc = 'upper left',bbox_to_anchor=(0.01, 0.99))
# handles = legend.legendHandles
# for i in range(len(handles)):
#     label = handles[i].get_label()
#     handles[i].set_alpha(0.5)


# ax1.set_xlim(t_min-(t_max-t_min)*0.05,t_max+(t_max-t_min)*0.05)
# plt.xticks(fontsize=20,fontproperties="Times New Roman")
# plt.yticks(fontsize=20,fontproperties="Times New Roman")
# plt.tight_layout()
# if save: 
#     plt.savefig('E:/磁参时序/forshow/t-prob/trainset/'+str(arnum_noaa)+'.jpg',bbox_inches='tight')

# if not imshow:
#     plt.clf()
#     plt.close('all') 
# # plt.show()



