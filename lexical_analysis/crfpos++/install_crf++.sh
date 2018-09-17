#!/bin/bash

# Based on Ubuntu

git clone https://github.com/taku910/crfpp.git;
pushd crfpp/;
./configure;
sed -i '/#include "winmain.h"/d' crf_test.cpp;
sed -i '/#include "winmain.h"/d' crf_learn.cpp;
sudo make install;
sudo ln -s crf_learn /usr/bin/crf_learn;
sudo ln -s crf_test /usr/bin/crf_test;
sudo ldconfig -v;
popd;
echo "done"
