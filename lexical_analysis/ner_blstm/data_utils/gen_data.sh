#!/bin/bash
head train_ner.txt -n 15677  > train_nert.txt;
tail train_ner.txt -n 3918 > dev_nert.txt;
mv train_nert.txt train_ner.txt;
head dev_nert.txt -n 1959 > dev_ner.txt;
tail dev_nert.txt -n 1959 > test_ner.txt;
rm dev_nert.txt;
