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
    train_file = 'train_pos.txt'
    dev_file = 'dev_pos.txt'
    test_file = 'test_pos.txt'
    cmd_tr = 'wget -P ../data/ https://raw.githubusercontent.com/Pelhans/pos_blstm/master/data/train_pos.txt'
    cmd_dev = 'wget -P ../dada/ https://raw.githubusercontent.com/Pelhans/pos_blstm/master/data/dev_pos.txt'
    cmd_te = 'wget -P ../dada/ https://raw.githubusercontent.com/Pelhans/pos_blstm/master/data/test_pos.txt'

    download(path, train_file, cmd_tr)
    download(path, dev_file, cmd_dev)
    download(path, test_file, cmd_te)

if __name__ == '__main__':
    main()
