#!/usr/bin/env python
# coding=utf-8

import commands

def transfer(file_name, output_name):
    with open(output_name, 'w') as o:
        with open(file_name) as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                line = line.strip('[\r\n\t]')
                words_tags = line.split(' ')
                total_words = len(words_tags)
                for word_num, word_tag in enumerate(words_tags):
                    if len(word_tag.split('/')) != 2:
                        print("A wrong format word_tag {} in line {}, skip it...".format(word_tag, line_num))
                        continue
                    word = word_tag.split('/')[0]
                    tag = word_tag.split('/')[1]
                    # remove title date like 19980101-01-001-001
                    if word.startswith('19980'):
                        continue
                    else:
                        o.write(word + '\t' + tag + '\n')
                    if  word_num == total_words - 2:
                        o.write('\n')

def seg_file(file_name):
    NR = commands.getoutput("sed -n '$=' {}".format(file_name))
    status_1, _ = commands.getstatusoutput("head -n {} {} > {} ".format(int(int(NR) * 0.8), file_name, 'train.data'))
    status_2, _ = commands.getstatusoutput("tail -n {} {} > {} ".format(int(int(NR) * 0.2), file_name, 'test.data'))
    if status_1 or status_2:
        print("Something worng here!")

if __name__ == '__main__':
    transfer('./people-daily.txt', 'output.data')
    seg_file('output.data')
