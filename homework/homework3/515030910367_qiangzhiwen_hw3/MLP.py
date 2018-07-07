import numpy as np
import argparse
import sys
from sklearn.neural_network import MLPClassifier
import random

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default="a3a/a3a", type=str, help='path of the training data')
parser.add_argument('--test_path', default="a3a/a3a.t", type=str,help='path of the testing data')
parser.add_argument('--features', default=123, type=int,help='number of features of the dataset')
"""
@param file: type{str} the path of the file to be read
@param features: the number of features each data consists
@return: this is a description of what is returned
@raise keyError: raises an exception
"""

def read_data(file,features,size):
    """read data form given path

    :param file: path of the file to be read
    :param features: number of features each data consists
    :param size: a float number between 0 and 1, determines the number of data to be trained in the file
    :return:
        data: train data, a n*m matrix, n data, each consists of m features
        label: label for training data, a n*1 matrix
    """
    f2 = open(file,"r")
    lines =np.array (f2.readlines())

    np.random.shuffle(lines)

    data_size=int(np.round (lines.shape[0]*size))
    #print ("data_size",data_size)
    data=np.zeros([data_size,features])
    label=np.zeros(data_size)

    for index,value in  enumerate(lines[:data_size]):
        data[index],label[index]=operating_lines(value,features)
    return [data,label]

def operating_lines(line,features):
    """change the format of each line

    :param line: the line of the original file
    :param features: number of features each data consists
    :return:
        data: a n*1 matrix, n represent the n features
        label: crrosponding labels of each data
    """
    line= line.split (' ')
    label=int(line[0])
    line=line[1:-1]
    data=np.zeros(features)
    for value in line:
        value=value.split(':')
        data[int(value[0])-1]=float(value[1])
    return [data,label]

def MLP(train_path="a3a/a3a",
        test_path="a3a/a3a.t",
        features=123,
        size=1,
        MLP_solver="adam",
        MLP_alpha=1e-5,
        MLP_hidden_layer_sizes=(200,100)
        ):
    """Using MLP to classify the data set

    :param train_path: the path of the training data
    :param test_path: the path of the testing data
    :param features: the number of features each data consists
    :param size: a float number between 0 and 1, determines the number of data to be trained in the file
    :param MLP_solver: The solver for weight optimization.
    :param MLP_alpha: L2 penalty (regularization term) parameter.
    :param MLP_hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    """
    train_data,train_label=read_data(train_path,features,size)
    #print ("train_data.shape",train_data.shape)
    #print ("train_label.shape",train_label.shape)

    test_data,test_label=read_data(test_path,features,1)
    clf = MLPClassifier(solver=MLP_solver, alpha=MLP_alpha,
                        hidden_layer_sizes=MLP_hidden_layer_sizes, random_state=1)

    clf.fit(train_data, train_label)
    predictions=clf.predict(test_data)
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_label))
    #print ("num_correct",num_correct)
    print ("The accuracy is %s"%(1.0*num_correct/len(test_label)))
    return (1.0*num_correct/len(test_label))

if __name__ == '__main__':
    MLP(train_path="w1a/w1a",test_path="w1a/w1a.t",features=300,size=0.4)
