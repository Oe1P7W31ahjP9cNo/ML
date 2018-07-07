# This program tests the performace of SVM on big data sets
import numpy as np
import time
from sklearn import svm
from sklearn.externals import joblib

def test_svm(save_model=0):
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
    train_img_normlization,train_label,test_img_normlization,test_label = get_CIFAR10_data()

    print("start training")
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.1,
                  decision_function_shape='ovr')
    clf.fit(train_img_normlization, train_label)
    print("finish training")
    if save_model==1:
        joblib.dump(clf, "CIFAR10_50000_train_model.m")

    predictions = [int(a) for a in clf.predict(test_img_normlization)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_label))
    for num in range(0,10):
        correctNum = sum(int(a ==y)&int(y==num) for a, y in zip(predictions, test_label))
        SumOfNum = sum(int(y==num) for y in test_label)
        print("category %s :%s of %s test values correct. The accuracy is %s."%(num,correctNum,SumOfNum,1.0*correctNum/SumOfNum))
    print ("%s of %s test values correct." % (num_correct, len(test_label)))
    print ("The accuracy is %s"%(1.0*num_correct/len(test_label)))
    print (time.strftime('%Y-%m-%d %H:%M:%S'))

# get the CIFAR10 data set
def get_CIFAR10_data():
    import cPickle
    with open('../cifar-10-batches-py/data_batch_1', 'rb') as fo:
        dict1 = cPickle.load(fo)
    with open('../cifar-10-batches-py/data_batch_2', 'rb') as fo:
        dict2 = cPickle.load(fo)
    with open('../cifar-10-batches-py/data_batch_2', 'rb') as fo:
        dict3 = cPickle.load(fo)
    with open('../cifar-10-batches-py/data_batch_2', 'rb') as fo:
        dict4 = cPickle.load(fo)
    with open('../cifar-10-batches-py/data_batch_2', 'rb') as fo:
        dict5 = cPickle.load(fo)
    with open('../cifar-10-batches-py/test_batch', 'rb') as fo:
        dict0 = cPickle.load(fo)
    train_data=np.concatenate((dict1['data'],dict2['data'],dict3['data'],dict4['data'],dict5['data']),axis=0)
    test_data=dict0['data']
    train_label=np.concatenate((np.array(dict1['labels']),np.array(dict2['labels']),
                                np.array(dict3['labels']),np.array(dict4['labels'])
                                ,np.array(dict5['labels'])),axis=0)
    test_label=np.array(dict3['labels'])
    return [np.round(train_data/255),train_label,np.round(test_data/255),test_label]

def main():
    test_svm()

if __name__ == "__main__":
    main()
