#!/bin/bash
# rbf svm-train -c 2.0 -g 0.03125 a3a.scale 
echo "begin"

# echo "Original sets with default parameters"
# svm-train a3a 
# svm-predict a3a.t a3a.model a3a.t.predict 

# echo "Scaled sets with default parameters"
# svm-scale -l -0 -u 1 -s range1 a3a > a3a.scale 
# svm-scale -r range1 a3a.t > a3a.t.scale 
# svm-train a3a.scale 
# svm-predict a3a.t.scale a3a.scale.model a3a.t.predict

echo "rbf grid test"
svm-train -c 2.0 -g 0.03125 a3a.scale 
svm-predict a3a.t.scale a3a.scale.model a3a.t.predict 

echo "linear"
svm-train -t 0 a3a.scale 
svm-predict a3a.t.scale a3a.scale.model a3a.t.predict 

echo "polynomial"
svm-train -t 1 a3a.scale 
svm-predict a3a.t.scale a3a.scale.model a3a.t.predict 

echo "sigmoid"
svm-train -t 3 a3a.scale 
svm-predict a3a.t.scale a3a.scale.model a3a.t.predict

echo "radial basis function"
svm-train -t 2 a3a.scale 
svm-predict a3a.t.scale a3a.scale.model a3a.t.predict 
echo "end"