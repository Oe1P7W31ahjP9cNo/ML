#-----------------------
#2017-10-23
#homework number logistic V1.1
#-----------------------
#现在进行逻辑回归对手写数字进行识别
#本代码基于扩充数据集进行运算，对比精度
#load datas
import pickle
import random
import sys
import time
import numpy as np
import gzip, struct

from scipy import io as spio
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def _read(image,label):
    
    with gzip.open(label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image,label

def get_data():
    
    train_img,train_label = _read(
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz')
    #print(train_img.shape)#得到训练集的大小
    
    test_img,test_label = _read(
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz')
    return [train_img,train_label,test_img,test_label]
    #return [test_img,test_label]#获取训练集信息

def operate_data():#二值化
    print("start loading")
    train_img,train_label,test_img,test_label = get_data()
    #归一化，调整为列向量
    train_img_normlization = np.reshape(np.round(train_img/255),(60000,-1))
    test_img_normlization = np.reshape(np.round(test_img/255),(10000,-1))
    #print( train_img_normlization.shape )
    #print( train_label.shape )
    print("finish loading")
    return [train_img_normlization,train_label,test_img_normlization,test_label]
    
def _logestic():
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
    train_img_normlization,train_label,test_img_normlization,test_label = operate_data()
    print("start training")
    model = LogisticRegression()
    model.fit( train_img_normlization,train_label)#进行学习
    print("finish training")
    print("start testing")
    predictions = model.predict(test_img_normlization)
    
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_label))
    for num in range(0,10):
        correctNum = sum(int(a ==y)&int(y==num) for a, y in zip(predictions, test_label))
        SumOfNum = sum(int(y==num) for y in test_label)
        print("number %s :%s of %s test values correct. The accuracy is %s."%(num,correctNum,SumOfNum,1.0*correctNum/SumOfNum))
    print ("%s of %s test values correct." % (num_correct, len(test_label)))
    print ("The accuracy is %s"%(1.0*num_correct/len(test_label)))
    print (time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    _logestic()


