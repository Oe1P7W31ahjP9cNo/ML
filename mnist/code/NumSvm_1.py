#-----------------------
#2017-10-17
#homework number svm V1
#-----------------------

#load datas
import pickle
import time
import numpy as np
import gzip, struct
from sklearn import svm
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
    print (train_img[1])
    return [train_img,train_label,test_img,test_label]
get_data()
def operate_data():#二值化
    train_img,train_label,test_img,test_label = get_data()
    #归一化，调整为行向量
    train_img_normlization = np.reshape(np.round(train_img/255),(60000,-1))
    test_img_normlization = np.reshape(np.round(test_img/255),(10000,-1))
    print( train_img_normlization.shape )
    print( train_label.shape )
    #print(train_img_normlization[0])
    return [train_img_normlization,train_label,test_img_normlization,test_label]

#operate_data()
def svm_baseline():
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
    train_img_normlization,train_label,test_img_normlization,test_label = operate_data()
    #training_data, validation_data, test_data =load_data()
    # 传递训练模型的参数，这里用默认的参数
    clf = svm.SVC()
    # clf = svm.SVC(C=8.0, kernel='rbf', gamma=0.00,cache_size=8000,probability=False)
    # 进行模型训练
    clf.fit(train_img_normlization, train_label)
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_img_normlization)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_label))
    print ("%s of %s test values correct." % (num_correct, len(test_label)))
    print (time.strftime('%Y-%m-%d %H:%M:%S'))

#if __name__ == "__main__":
#    svm_baseline()
    
"""
# 将 28px * 28px 的图像数据转换成 1*28^2 的 numpy 向量
# 参数：imgFile--图像名  如：0_1.png
# 返回：1*400 的 numpy 向量
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i') # 20px * 20px 灰度图像
    img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normlization, (1,-1)) # 1 * 400 矩阵
    return img_arr2



def load_data():
    file = gzip.open('train-images-idx3-ubyte.gz', 'rb')
    pickle.load(file)
    file.close()
    #return (training_data, validation_data, test_data)

#SVM main code
from sklearn import svm
#import time


load_data()
"""
'''

def svm_baseline():
    #print time.strftime('%Y-%m-%d %H:%M:%S') 
    training_data, validation_data, test_data =load_data()
    # 传递训练模型的参数，这里用默认的参数
    clf = svm.SVC()
    # clf = svm.SVC(C=8.0, kernel='rbf', gamma=0.00,cache_size=8000,probability=False)
    # 进行模型训练
    clf.fit(training_data[0], training_data[1])
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "%s of %s test values correct." % (num_correct, len(test_data[1]))
    #print time.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    svm_baseline()
'''
