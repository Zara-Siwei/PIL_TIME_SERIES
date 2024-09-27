# import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM,Conv1D,Dropout,Bidirectional,Multiply
from tensorflow.keras.models import Model

# from keras.layers.merging.concatenate import concatenate
from tensorflow.keras.layers import add,multiply,concatenate,average,dot


# import sys
# sys.path.append('E:/磁参时序/cnn-bilstm-attention-time-series-prediction_keras-master')  # 文件夹的绝对路径
# from attention_utils import get_activations

from tensorflow.keras import layers

from tensorflow.python.keras.layers.core import *
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import *

import pandas as pd
import numpy as np






SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = layers.Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = layers.Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul



# def create_dataset(dataset, look_back):
#     '''
#     对数据进行处理
#     '''
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back),:]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back,:])
#     TrainX = np.array(dataX)
#     Train_Y = np.array(dataY)

#     return TrainX, Train_Y

#多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data, normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data


def attention_model():
    inputs = layers.Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = layers.Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    x = layers.Dropout(0.3)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_out = Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = layers.Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = layers.Flatten()(attention_mul)

    output = layers.Dense(2, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model




for fold_selected in [8,9]:
    
    # 加载数据 
    t_set1s = [6,12,18,24,30,36,42,48]
    
    for t_set1 in t_set1s:
        # t_set1 = 48
        
        INPUT_DIMS = 18
        TIME_STEPS = t_set1+1
        lstm_units = 64
        
        # fold_selected = 1
        trange_sq = t_set1
        datatype=f"now_{trange_sq}h"
        dataPath = 'H:/dataset/pil/' + 'dataset_2class_'+datatype+'_sq/'
        adv = 24
        para_train = np.load(dataPath  + str(adv)+f'/para_train_{fold_selected}.npy')
        label_train = np.load(dataPath  + str(adv)+f'/label_train_{fold_selected}.npy')
        # para_test = np.load(dataPath  + str(adv)+'/para_test.npy')
        # label_test = np.load(dataPath  + str(adv)+'/label_test.npy')
        # para_test = np.load(dataPath  + str(adv)+'/para_data_test_unbal.npy')
        # label_test = np.load(dataPath  + str(adv)+'/label_data_test_unbal.npy')
        
        # label_test = abs(label_test)
        label_train = abs(label_train)
        
        
        para_2d_train = para_train.reshape((para_train.shape[0]*para_train.shape[1],para_train.shape[2]))
        # para_2d_test = para_test.reshape((para_test.shape[0]*para_test.shape[1],para_test.shape[2]))
        
        #归一化
        # print(para_train.shape) 
        from sklearn.preprocessing import MinMaxScaler
        scalarX = MinMaxScaler()
        scalarX.fit(para_2d_train)
        # scalarY.fit(y_train)
        x_train = scalarX.transform(para_2d_train)
        # y = scalarY.transform(y_train)
        
        # x_test = scalarX.transform(para_2d_test)
        
        
        import joblib
        joblib.dump(scalarX,dataPath +str(adv)+'/scale_'+f'minmax_fold{fold_selected}'+'.save') 
        # scalarX = joblib.load(dataPath +str(adv)+'/scale_'+'minmax'+'.save') 
        
        
        x_train = x_train[:,2::]
        # x_test = x_test[:,2::]
        
        x_train = x_train.reshape((para_train.shape[0],para_train.shape[1],x_train.shape[1]))
        # x_test = x_test.reshape((para_test.shape[0],para_test.shape[1],x_test.shape[1]))
        
        # pollution_data = data[:,0].reshape(len(data),1);
        
        
        
        from keras.utils import np_utils
        # from sklearn.model_selection import train_test_split, KFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        
        
        encoder = LabelEncoder()
        encoded_Y = encoder.fit_transform(label_train)
        # encoded_Ytest = encoder.transform(label_test)
        
        # convert integers to dummy variables (one hot encoding)
        
        dummy_y = np_utils.to_categorical(encoded_Y)
        # dummy_ytest = np_utils.to_categorical(encoded_Ytest)
        
        uniques, ids = np.unique(encoded_Y, return_inverse=True)
        
        # print(train_X.shape,train_Y.shape)
        
        m = attention_model()
        # m.summary()
        m.compile(optimizer='adam', loss='categorical_crossentropy')
        
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        modeltype = 'cnn-lstm-at_'+datatype+f'_fold{fold_selected}'
        save_checkpoint = dataPath  + str(adv) + "/check_{epoch:02d}_{loss:.4f}_"+modeltype+".hdf5"
        checkpoint = ModelCheckpoint(save_checkpoint, monitor='loss', verbose=1, save_best_only=True, mode='min',period=10)
        callbacks_list = [checkpoint]
        
        history = m.fit([x_train], dummy_y, epochs=100, batch_size=40, callbacks=callbacks_list,
                  
              # validation_split=0.1
              )
        
        
        
        # m.save("./model.h5")
        # np.save("normalize.npy",normalize)
        
        
        
        m.save(dataPath +str(adv)+'/model_'+modeltype+'.h5')
        
        
        # # from tensorflow.keras.models import load_model
        # # file_pre = dataPath  + str(adv) + "/check_10_0.2132_cnn-lstm-at-ppt.hdf5"
        # # m = load_model(file_pre)
        
        # pred = m.predict(x_test)
        # pred = uniques[pred.argmax(1)]
        # y_pred = encoder.inverse_transform(pred)
        
        # from sklearn.metrics import classification_report  
        # res = classification_report(label_test,y_pred,labels=[0,1],target_names=['非强耀斑','强耀斑'])
        # print(str(adv) + '小时的提前量下，预报的结果:')
        # print(res)
        
        # get_activations(m,layers.Dense())
        
        import matplotlib.pyplot as plt
        
        plt.figure(dpi=300)
        plt.plot(history.history['loss'])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train"],loc="upper left")
        plt.savefig(dataPath +str(adv)+'/'+modeltype+f'100epoch_fold{fold_selected}.png',dpi=300)
    
    


# # # =============================================================================
# # # 后续分析 ROC曲线
# # # =============================================================================


# m = load_model(dataPath  + str(adv) + '/check_10_0.2150_cnn-lstm-at-now.hdf5')


# pred = m.predict(x_test)
# pred_01 = pred.argmax(1)
# y_pred = pred_01
# # y_pred = encoder.inverse_transform(pred)




# import numpy as np
# from sklearn.metrics import roc_auc_score,roc_curve,auc
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt



# def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=300):
#         """
#         将多个机器模型的roc图输出到一张图上
       
#         Args:
#             names: list, 多个模型的名称
#             sampling_methods: list, 多个模型的实例化对象
#             save: 选择是否将结果保存（默认为png格式）
           
#         Returns:
#             返回图片对象plt
#         """
#         plt.figure(dpi=dpin)

#         for (name, method, colorname) in zip(names, sampling_methods, colors):
#             titlesize = 20
#             labelsize = 20
#             axissize = 20
#             lw = 3
            
           
            
#             y_test_preds = method.predict(X_test)
#             # y_test_predprob = method.predict_proba(X_test)[:,1]
#             y_test_predprob = method.predict_on_batch(X_test)[:,1]
#             fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
           
#             plt.rcParams['mathtext.default'] = 'regular'
#             plt.rcParams['font.sans-serif'] = "Times New Roman"
#             plt.rcParams['font.size'] = axissize
           
#             plt.plot(fpr, tpr, lw=lw, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)),color = colorname)
#             plt.plot([0, 1], [0, 1], linestyle='dashdot', lw=lw, color = 'grey')
#             plt.axis('square')
#             plt.xlim([0, 1])
#             plt.ylim([0, 1])
#             plt.yticks([0.2,0.4,0.6,0.8,1.0])
#             # plt.set_ylabel('T\n(K)',fontproperties="Times New Roman",size=size_label)
 
#             font = {'family':'Times New Roman'  #'serif', 
#                     #         ,'style':'italic'
#                     ,'weight':'normal'
#                     #         ,'color':'red'
#                     ,'size':10
#                   }

#             plt.xlabel('False Positive Rate',fontsize=labelsize,fontproperties="Times New Roman")
#             plt.ylabel('True Positive Rate',fontsize=labelsize,fontproperties="Times New Roman")
#             plt.title('ROC Curve',fontsize=titlesize)
#             plt.legend(prop=font
#                       ,loc = 'lower right'
#                       ,bbox_to_anchor=(1, 0.0)   # (x, y, width, height) (0, 0.5, 0.5, 0.5)
#                       # , bbox_to_anchor=(0.5, 0.5, 0.5, 0.5)
#                       , markerscale = 1.5 # legend里面的符号的大小
#                       )
#             # plt.legend(loc='lower right',fontsize=30)
#             plt.show()
#         if save:
#             plt.savefig('models_roc.png')
#         return plt


# # plt.setp(patches, 'facecolor', 'tomato', 'alpha', 0.75)
# # plt.rcParams['mathtext.default'] = 'regular'
# # plt.rcParams['font.sans-serif'] = "Times New Roman"
# # plt.rcParams['font.size'] = 12
# # # ax1.set_xlim(-max(max(bins1),max(bins2))*0.05,max(max(bins1),max(bins2))*1.05)
# # settings1.append(s1)
# # settings1.append(s2)
# # settings1.append(scale)
# # print(s1,s2,scale)
# # font = {'family':'Times New Roman'  #'serif', 
# #          #         ,'style':'italic'
# #         ,'weight':'normal'
# #          #         ,'color':'red'
# #         ,'size':12
# #        }

# # key = patches+h
# # labs = [l.get_label() for l in key]
# # ax1.legend(key,labs,prop=font
# #            ,loc = 'upper left'
# #            ,bbox_to_anchor=(0.56, 0.98)   # (x, y, width, height) (0, 0.5, 0.5, 0.5)
# #            # , bbox_to_anchor=(0.5, 0.5, 0.5, 0.5)
# #            , markerscale = 1.5 # legend里面的符号的大小
# #            )



# names = ['CNN-BiLSTM-AT']
# # names = ['Decision Tree', # 这个就是模型标签，我们使用三个，所以需要三个标签
# #          'Naive Bayes',
# #          'knn']

# sampling_methods = [m]
# # sampling_methods = [clf_d, # 这个就是训练的模型。
# #                     gnb,
# #                     knn
# #                    ]

# colors = ['orange']
# # colors = ['crimson',  # 这个是曲线的颜色，几个模型就需要几个颜色哦！
# #           'orange',
# #           'lawngreen'
# #          ]

# #ROC curves
# test_roc_graph = multi_models_roc(names, sampling_methods, colors, x_test, label_test, save = False)  # 这里可以改成训练集





























