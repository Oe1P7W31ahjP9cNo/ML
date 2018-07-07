import MLP
import numpy as np
size=[0.1,0.5,0.8,1]
array=np.zeros([4,4])

def test_w1a_data():
    for index,value in enumerate(size):
        array[1][index]=MLP.MLP(train_path="w1a/w1a",test_path="w1a/w1a.t",features=300,size=value)

def test_a3a_data():
    for index,value in enumerate(size):
        array[0][index]=MLP.MLP(train_path="a3a/a3a",test_path="a3a/a3a.t",features=123,size=value)
def test_svmguide1_data():
    for index,value in enumerate(size):
        array[2][index]=MLP.MLP(train_path="svmguide1/svmguide1.scale",test_path="svmguide1/svmguide1.t.scale",features=4,size=value)
   
def test_pendigits_data():
    for index,value in enumerate(size):
        array[3][index]=MLP.MLP(train_path="pendigits/pendigits.scale",test_path="pendigits/pendigits.t.scale",features=16,size=value)


def main():
    print "w1a begin"
    test_w1a_data()
    print "w1a end"
    
    print "a3a begin"
    test_a3a_data()
    print "a3a end"

    print "svmguide1 begin"
    test_svmguide1_data()
    print "svmguide1 end"

    print "pendigits begin"
    test_pendigits_data()
    print "pendigits end"
    print array

if __name__ == "__main__":
    main()



