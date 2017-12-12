#coding=utf8




import sys

#home_dir = "D:/source/NLP/people_daily//"

home_dir = "./"

def saveDataFile(trainobj,testobj,isTest,word,handle):
    if isTest:
        saveTrainFile(testobj,word,handle)
    else:
        saveTrainFile(trainobj,word,handle)

def saveTrainFile(fiobj,word,handle): 
    if len(word) > 0 and  word != "。" and word != "，":
        fiobj.write(word + '\t' + handle  + '\n')
    else:
        fiobj.write('\n')

def convertTag():    
    fiobj    = open( home_dir + 'people-daily.txt','r')
    trainobj = open( home_dir +'train.data','w' )
    testobj  = open( home_dir  +'test.data','w')

    arr = fiobj.readlines()
    i = 0
    for a in sys.stdin:
        i += 1
        a = a.strip('\r\n\t ')
        if a=="":continue
        words = a.split(" ")
        test = False
        if i % 10 == 0:
            test = True
        for word in words[1:]:
            print "---->", word
            word = word.strip('\t ')
            if len(word) > 0:        
                i1 = word.find('[')
            if i1 >= 0:
                word = word[i1+1:]
            i2 = word.find(']')
            if i2 > 0:
                w = word[:i2]
            word_hand = word.split('/')
            print "----",word
            w,h = word_hand
            #print w,h
            if h == 'nr':    #ren min
                #print 'NR',w
                if w.find('·') >= 0:
                    tmpArr = w.split('·')
                    for tmp in tmpArr:
                        saveDataFile(trainobj,testobj,test,tmp,h)
                    continue
            saveDataFile(trainobj,testobj,test,w,h)
        saveDataFile(trainobj, testobj, test,"","")
            
    trainobj.flush()
    testobj.flush()

if __name__ == '__main__':    
    convertTag()


