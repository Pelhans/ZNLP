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
    file_ner = codecs.open(filename + "_fix.txt", "w", encoding = "utf-8")
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
        file_ner.write(line)
        #print line
        #print line[330:344]

def change_tag(filename="train"):
    '''
    Change POS tag to NER tag.
    Only nz nt ns nr nan survived.
    '''
    sentences = []
    useful_tag = ["nz", "nt", "ns", "nr"]
    file = codecs.open(filename + "_fix.txt", "r", encoding="utf-8")
    file_ner = codecs.open(filename +"_ner.txt", "w", encoding="utf-8")
    for line in file:
        words = []
        word = []
        tag = []
        new_word = []
        num_nan = 0
        wordsplit = re.split("\s+", line.replace("\n", ""))
        words.extend(wordsplit)
        #print words
        for word_tag_pair in words:
            pairs = word_tag_pair.split("/")
            if len(pairs)==2:
                if len(pairs[0].strip()) != 0 and len(pairs[1].strip()) != 0 :
                    word.append(pairs[0])
                    tag.append(pairs[1])
        for i in range(len(tag)):
            if tag[i] not in useful_tag:
                tag[i] = "nan"
        for i in range(len(tag)):
            if tag[i] == "nan":
                num_nan += 1
        for i in range(len(tag)):
            new_word.append(word[i] + "/" + tag[i])
        #print new_word
        new_line = u" ".join(new_word)
        if len(tag) > 0:
            if (1 - (float(num_nan) / len(tag))) >= 0.2:
                file_ner.write("%s\n" %(new_line))

def main():
    if not os.path.exists("../data/train_fix.txt"):
        print "Producing fixed_pos file....."
        _read_file("../data/train")

    if not os.path.exists("../data/train_ner.txt"):
        print "Producing ner file....."
        change_tag("../data/train")

if __name__ == "__main__":
    main()
