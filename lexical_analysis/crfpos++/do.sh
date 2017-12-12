#crf_learn -f 3 -p 40 -c 4.0 template train.data model > train.rst  
crf_test -m model test.data > test.rst

