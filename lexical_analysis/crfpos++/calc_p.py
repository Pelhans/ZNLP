#!/usr/bin/env python
# coding=utf-8
import sys
import commands

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename) as f:
        total_tag = 0
        correct_tag = 0
        for line_num in range(int(commands.getoutput("sed -n '$=' {}".format(filename)))):
            line = f.readline().strip()
            if line == '':
                continue
            word, gold_tag, pre_tag = line.split()
            total_tag += 1
            if gold_tag == pre_tag:
                correct_tag += 1
        print('Total tag: {}\nThe accuracy is {}'.format(total_tag, correct_tag/float(total_tag)))

