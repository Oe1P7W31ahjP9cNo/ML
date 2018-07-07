#!/bin/bash
#8.0 2.0 99.6664
echo "test pendigits begin"

# echo "Original sets with default parameters begin"
# svm-train pendigits 
# svm-predict pendigits.t pendigits.model pendigits.t.predict 
# echo "Original sets with default parameters end"

# echo "Scaled sets with default parameters begin"
# svm-scale -l -1 -u 1 -s range1 pendigits > pendigits.scale 
# svm-scale -r range1 pendigits.t > pendigits.t.scale 
# svm-train pendigits.scale 
# svm-predict pendigits.t.scale pendigits.scale.model pendigits.t.predict
# echo "Scaled sets with default parameters end"

# echo "linear kernel begin"
# svm-train -t 0 pendigits.scale 
# svm-predict pendigits.t.scale pendigits.scale.model pendigits.t.predict 
# echo "linear kernel end"

# echo "polynomial kernel begin"
# svm-train -t 1 pendigits.scale 
# svm-predict pendigits.t.scale pendigits.scale.model pendigits.t.predict 
# echo "polynomial kernel end"

# echo "sigmoid kernel begin"
# svm-train -t 3 pendigits.scale 
# svm-predict pendigits.t.scale pendigits.scale.model pendigits.t.predict
# echo "sigmoid kernel end"

# echo "radial basis function with default parameters begin"
# svm-train -t 2 pendigits.scale 
# svm-predict pendigits.t.scale pendigits.scale.model pendigits.t.predict 
# echo "radial basis function with default parameters end"

echo "radial basis function with grid test begin"
svm-train -c 8.0 -g 2.0 pendigits.scale 
svm-predict pendigits.t.scale pendigits.scale.model pendigits.t.predict 
echo "radial basis function with grid test end"

echo "test pendigits end"
