#!/usr/bin/env python
# coding=utf-8


import re
import os
import sys
import time
import codecs
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from itertools import chain

def _read_file(filename = "../data/train"):
    file = codecs.open(filename + ".txt", encoding='utf-8')
    file_pos = codecs.open(filename + "_fpos.txt", "w", encoding = "utf-8")
    for line in file:
        #print line
        leftB = []
        rightB = []
        replace_word = []
        record_word = []
        len_init_word = []
        if u'[' in line:
            #print "ha"
            if u']' in line:
                #print "ha"
                for i in range(len(line)):
                    if line[i] == '[':
                        leftB.append(i)
                    if line[i] == ']':
                        rightB.append(i)
        #print leftB, rightB
        #print line[leftB[2] : rightB[2]]
        for i in range(len(leftB)):
            new_word = []
            word_tag_paireeee = re.split('  ', line[leftB[i]+1:rightB[i]])
           # print word_tag_paireeee[1].split("/")[0]
            word_tag_paireeee = np.asarray(word_tag_paireeee)
            #print np.shape(word_tag_paireeee)
            len_init_word.append(len(word_tag_paireeee))
            for j in range(len(word_tag_paireeee)):
                new_word.append(word_tag_paireeee[j].split("/")[0])
            replace_word.append(u"".join(new_word[:]))
            
        #print replace_word[:len(word_tag_paireeee)]
        #print len(replace_word[0]), 'I got full words here'
        for i in range(len(rightB)):
            len_remove = 0
            len_replace = 0
            for j in range(0, i):
                len_remove  += rightB[j] - leftB[j]  #- 2*(len_init_word[j-1] -1)
                len_replace += len(replace_word[j])
            leftB[i] = leftB[i] - len_remove + len_replace 
            rightB[i] = rightB[i] - len_remove + len_replace
            #print line[leftB[i]:rightB[i]]
            line = line[:leftB[i]] + replace_word[i] +  "/" +line[rightB[i]+1:]
        file_pos.write(line)
        #print line
        #print line[330:344]


def main():
    #Clean data to remove word in [  ]
    _read_file("../data/train")

if __name__ == "__main__":
    main()
