#!/bin/bash
echo "begin"
# echo "Original sets with default parameters"
# svm-train a1a 
# svm-predict a1a.t a1a.model a1a.t.predict 

# echo "Scaled sets with default parameters"
# svm-scale -l -0 -u 1 -s range1 a1a > a1a.scale 
# svm-scale -r range1 a1a.t > a1a.t.scale 
# svm-train a1a.scale 
# svm-predict a1a.t.scale a1a.scale.model a1a.t.predict

echo "Scaled sets with parameter selection"
svm-train -c 512.0 -g 0.0001220703125 a1a.scale 
svm-predict a1a.t.scale a1a.scale.model a1a.t.predict 
echo "end"