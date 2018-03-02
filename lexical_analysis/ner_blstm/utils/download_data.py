#!/usr/bin/env python
# coding=utf-8

import os
import subprocess


def download(path, filename, cmd):
    log = open(path + filename + '_log.txt', 'w')
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+filename):
        status = subprocess.call(cmd, shell = True)
        if status != 0:
            log.write('\nFailed:' +filename)
            #continue
        log.write('\nSucess: ' + filename)
    log.flush()
    log.close

def main():
    path = "../data/"
    train_file = 'train_ner.txt'
    dev_file = 'dev_ner.txt'
    test_file = 'test_ner.txt'
    cmd_tr = 'wget -P ../data/ https://raw.githubusercontent.com/Pelhans/ner_blstm/master/data/train_ner.txt'
    cmd_dev = 'wget -P ../dada/ https://raw.githubusercontent.com/Pelhans/ner_blstm/master/data/dev_ner.txt'
    cmd_te = 'wget -P ../dada/ https://raw.githubusercontent.com/Pelhans/ner_blstm/master/data/test_ner.txt'

    download(path, train_file, cmd_tr)
    download(path, dev_file, cmd_dev)
    download(path, test_file, cmd_te)

if __name__ == '__main__':
    main()
