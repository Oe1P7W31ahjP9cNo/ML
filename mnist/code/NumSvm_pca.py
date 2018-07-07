#-----------------------
#2017-10-17
#homework number svm +PCA
#-----------------------
#结合PCA进行降维运算
#load datas
import pickle
import sys
import time
import numpy as np
import gzip, struct
from sklearn import svm
from sklearn.decomposition import PCA
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
    test_img,test_label = _read(
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz')
    return [train_img,train_label,test_img,test_label]


def operate_data_PCA():#降维+二值化
    #pca = decomposition.PCA(n_components=3)
    pca=PCA(n_components=700)#降维成400
    print("start loading")
    train_img,train_label,test_img,test_label = get_data()
    #归一化，调整为列向量
    train_img_normlization = np.reshape(np.round(train_img/255),(60000,-1))
    test_img_normlization = np.reshape(np.round(test_img/255),(10000,-1))
    new_decline_train_img=pca.fit_transform(train_img_normlization[1000:10000])
    new_decline_test_img = pca.fit_transform(test_img_normlization)
    #print( train_img_normlization.shape )
    #print( train_label.shape )
    print("finish loading")
    return [ new_decline_train_img,train_label[1000:10000],new_decline_test_img,test_label]
    
        
def svm_baseline():
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
    #train_img_normlization,train_label,test_img_normlization,test_label = operate_data()
    train_img_normlization,train_label,test_img_normlization,test_label = operate_data_PCA()
    #clf = svm.SVC() # 默认的参数
    
    print("start training")
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.004,
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


