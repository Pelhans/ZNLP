#!/bin/bash
#!/usr/bin/env python

#python gen_pos_file.py    #geposate pos file from PFR POS corpus
python get_pickle.py      #for train dataset
python get_pickle_dev.py  #for valid dataset
python get_pickle_test.py #for test dataset
