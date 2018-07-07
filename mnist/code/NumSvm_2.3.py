#-----------------------
#2017-10-23
#homework number svm V2.3
#-----------------------
#现在进行旋转图像，从而扩充数据集
#本代码基于扩充数据集进行运算，对比精度
#load datas
import pickle
import random
import sys
import time
import numpy as np
import gzip, struct
from sklearn import svm
from PIL import Image #用来处理图像的库，旋转，缩放等
def _read(image,label):
    
    with gzip.open(label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image,label

def get_data():
    '''
    train_img,train_label = _read(
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz')
    #print(train_img.shape)#得到训练集的大小
    '''
    test_img,test_label = _read(
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz')
    #return [train_img,train_label,test_img,test_label]
    return [test_img,test_label]#获取训练集信息


def enlarge_Rotate_data():#通过旋转的手段，对训练集进行扩充,扩充后的数据集为原来的一倍
    train_img,train_label = _read(
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz')
    #获取训练集信息，此处就不调用get_data()了
    #创建三维数组
    new_im_rotate = np.zeros((60000,28,28))
    
    for i in range(0,60000):#旋转思路：通过PIL中的函数进行旋转，但需要先将矩阵变成图片才能导入
        data = train_img[i]
        new_im = Image.fromarray(data.astype(np.uint8))
        new_im = new_im.rotate(random.randint(-15, +15))#每张图片随机旋转指定范围内的角度
        new_data = np.asarray(new_im)
        new_im_rotate[i] = new_data
        
    #data = train_img[1]
    #print(matrix)
    new_im_rotate_add = np.append(train_img,new_im_rotate,axis=0)#与原数据集进行拼接
    new_train_label_add = np.append(train_label,train_label,axis=0)#double 标签集
    print(new_im_rotate_add.shape)#由此可知，在原有数据集中每个图像是按28*28的矩阵存储的
    return [new_im_rotate_add, new_train_label_add]

def operate_data():#二值化
    print("start loading")
    train_img,train_label = enlarge_Rotate_data()
    test_img,test_label = get_data()
    #归一化，调整为列向量
    train_img_normlization = np.reshape(np.round(train_img/255),(120000,-1))
    test_img_normlization = np.reshape(np.round(test_img/255),(10000,-1))
    #print( train_img_normlization.shape )
    #print( train_label.shape )
    print("finish loading")
    return [train_img_normlization,train_label,test_img_normlization,test_label]
    
        
def svm_baseline():
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
    train_img_normlization,train_label,test_img_normlization,test_label = operate_data()
    #clf = svm.SVC() # 默认的参数
    
    print("start training")
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03,
                  decision_function_shape='ovr')#‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # 进行模型训练
    clf.fit(train_img_normlization, train_label)
    print("finish training")
    
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_img_normlization)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_label))
    for num in range(0,10):
        correctNum = sum(int(a ==y)&int(y==num) for a, y in zip(predictions, test_label))
        SumOfNum = sum(int(y==num) for y in test_label)
        print("number %s :%s of %s test values correct. The accuracy is %s."%(num,correctNum,SumOfNum,1.0*correctNum/SumOfNum))
    print ("%s of %s test values correct." % (num_correct, len(test_label)))
    print ("The accuracy is %s"%(1.0*num_correct/len(test_label)))
    print (time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    svm_baseline()


