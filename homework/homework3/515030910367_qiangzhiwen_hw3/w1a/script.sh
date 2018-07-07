#!/bin/bash
#128.0 0.0078125 98.3044
echo "begin"

# echo "Original sets with default parameters"
# svm-train w1a 
# svm-predict w1a.t w1a.model w1a.t.predict 

# echo "Scaled sets with default parameters"
# svm-scale -l -0 -u 1 -s range1 w1a > w1a.scale 
# svm-scale -r range1 w1a.t > w1a.t.scale 
# svm-train w1a.scale 
# svm-predict w1a.t.scale w1a.scale.model w1a.t.predict
echo "rbf grid test"
svm-train -c 128.0 -g 0.0078125 w1a.scale 
svm-predict w1a.t.scale w1a.scale.model w1a.t.predict 
echo "rbf grid test end"

echo "linear"
svm-train -t 0 w1a.scale 
svm-predict w1a.t.scale w1a.scale.model w1a.t.predict 
echo "linear end"

echo "polynomial"
svm-train -t 1 w1a.scale 
svm-predict w1a.t.scale w1a.scale.model w1a.t.predict 
echo "polynomial end"

echo "sigmoid"
svm-train -t 3 w1a.scale 
svm-predict w1a.t.scale w1a.scale.model w1a.t.predict
echo "sigmoid end"

echo "radial basis function"
svm-train -t 2 w1a.scale 
svm-predict w1a.t.scale w1a.scale.model w1a.t.predict 
echo "radial basis function end"
echo "end"