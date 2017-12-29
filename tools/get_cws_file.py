# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2017-01-25 11:46:37
# @Last Modified by:   Pelhans
# @Last Modified time: 2017-12-29 09:47:31

import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8') 
totalLine = 0
longLine = 0
maxLen = 80


def processToken(token, collect, out, endn):
  global totalLine
  global longLine
  global maxLen
  nn = len(token)
  while nn > 0 and token[nn - 1] != '/':
    nn = nn - 1

  token = token[:nn - 1].strip()
  if not token:
    return
  token = str(token).decode("utf-8")
  if len(token) == 1:
    token = token + "/s  "
    out.write("%s" % (token))
  elif len(token) ==2:
    begin = token[0] + "/b  "
    out.write("%s" % (begin))
    end = token[1] + "/e  "
    out.write("%s" % (end))
  elif len(token) >2:
    begin = token[0] + "/b  "
    out.write("%s" % (begin))
    for i in range(1, len(token)):
        mid = token[i] + "/m  "
        out.write("%s" % (mid))
    end = token[-1] + "/e  "
    out.write("%s" % (end))
  if endn:
    out.write("\n")


def processLine(line, out):
  line = line.strip()
  nn = len(line)
  seeLeftB = False
  start = 0
  collect = []
  try:
    for i in range(nn):
      if line[i] == ' ':
        if not seeLeftB:
          token = line[start:i]
          if token.startswith('['):
            tokenLen = len(token)
            while tokenLen > 0 and token[tokenLen - 1] != ']':
              tokenLen = tokenLen - 1
            token = token[1:tokenLen - 1]
            ss = token.split(' ')
            for s in ss:
              processToken(s, collect, out, False)
          else:
            processToken(token, collect, out, False)
          start = i + 1
      elif line[i] == '[':
        seeLeftB = True
      elif line[i] == ']':
        seeLeftB = False
    if start < nn:
      token = line[start:]
      if token.startswith('['):
        tokenLen = len(token)
        while tokenLen > 0 and token[tokenLen - 1] != ']':
          tokenLen = tokenLen - 1
        token = token[1:tokenLen - 1]
        ss = token.split(' ')
        ns = len(ss)
        for i in range(ns - 1):
          processToken(ss[i], collect, out, False)
        processToken(ss[-1], collect, out, True)
      else:
        processToken(token, collect, out, True)
  except Exception as e:
    pass


def main(argc, argv):
  global totalLine
  global longLine
  if argc < 3:
    print("Usage:%s <dir> <output>" % (argv[0]))
    sys.exit(1)
  file = argv[1]
  out = open(argv[2], "w")
  if file.endswith(".txt"):
    # print("processing:%s" % (curFile))
    fp = open(file, "r")
    for line in fp.readlines():
        line = line.strip()
        processLine(line, out)
    fp.close()
  else:
      print ("Please use .txt file as input!!")
  out.close()
  print("total:%d, long lines:%d" % (totalLine, longLine))


if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
