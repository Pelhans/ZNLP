#!/bin/bash
head train_fpos.txt -n 15677  > train_post.txt;
tail train_fpos.txt -n 3918 > dev_post.txt;
cp train_fpost.txt train_pos.txt;
head dev_post.txt -n 1959 > dev_pos.txt;
tail dev_post.txt -n 1959 > test_pos.txt;
rm dev_post.txt;
