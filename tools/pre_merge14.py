#!/usr/bin/env python
# coding=utf-8

import sys ,os
import time

totalLine = 0
longLine = 0
maxLen = 80

def main(argc, argv):
    '''
    Code for PFR2014
    '''
    global totalLine
    global longLine
    if argc < 3:
        print("Usage:%s <dir> <output>" % (argv[0]))
        sys.exit(1)
    start = time.time()
    rootDir = argv[1]
    out = open(argv[2], "w+")
    for dirName, subdirList, fileList in os.walk(rootDir):
        curDir = os.path.join(rootDir, dirName)
        for file in fileList:
            if file.endswith(".txt"):
                curFile = os.path.join(curDir, file)
                print("processing:%s" % (curFile))
                fp = open(curFile, "r")
                out.writelines(fp.readlines())
                #for line in fp.readlines():
                    #line = line.strip()
                #    if not line:
                #        out.write(str(line))
                fp.close()
    out.close()
    print ("time: %s s"% (str(time.time() - start)))

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

