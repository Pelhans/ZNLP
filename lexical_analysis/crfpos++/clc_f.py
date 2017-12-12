#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import sys
 
if __name__=="__main__":
    try:
        file = open(sys.argv[1], "r")
    except:
        print "result file is not specified, or open failed!"
        sys.exit()
    wc = 0
    wc_of_test = 0
    wc_of_gold = 0
    wc_of_correct = 0
    flag = True
    
    for l in file:
        if l=='\n': continue
    
        _, g, r = l.strip().split()
     
        if r != g:
            flag = False
   	wc += 1
	 
        if flag:
            wc_of_correct +=1
        flag = True
    
 
    print "WordCount from result:", wc
    print "WordCount of correct post :", wc_of_correct
            
    #准确率
    P = wc_of_correct/float(wc)
    
    print "准确率:%f" % (P)

