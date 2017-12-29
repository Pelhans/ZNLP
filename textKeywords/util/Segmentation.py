#-*- encoding:utf-8 -*-


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import jieba.posseg as pseg
import codecs
import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]

import util


def get_default_stop_words_file():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, '../data/stopwords.txt')

class WordSegmentation(object):
    """ 分词 """
    
    def __init__(self, stop_words_file = None, allow_speech_tags = util.allow_speech_tags):
        """
        Keyword arguments:
        stop_words_file    -- 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
        allow_speech_tags  -- 词性列表，用于过滤
        """     
        
        allow_speech_tags = [util.as_text(item) for item in allow_speech_tags]

        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = set()
        self.stop_words_file = get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.stop_words_file = stop_words_file
        for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
            self.stop_words.add(word.strip())
    
    def segment(self, pic , pipe, text, lower = True, use_stop_words = True, use_speech_tags_filter = False, use_jieba = False):
        """对一段文本进行分词，返回list类型的分词结果

        Keyword arguments:
        lower                  -- 是否将单词小写（针对英文）
        use_stop_words         -- 若为True，则利用停止词集合来过滤（去掉停止词）
        use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。    
        """
        text = util.as_text(text).split("，")
        if len(text)> 0:
            text= text[0]
        if use_jieba:
            jieba_result = pseg.cut(text)
            if use_speech_tags_filter == True:
                jieba_result = [w for w in jieba_result if w.flag in self.default_speech_tag_filter]
            else:
                jieba_result = [w for w in jieba_result]

            # 去除特殊符号
            word_list = [w.word.strip() for w in jieba_result if w.flag!='x']
            word_list = [word for word in word_list if len(word)>0]
        else:
            word2id_c, id2tag_c, word2id_p, id2tag_p, word2id_n, id2tag_n, zy = pic
            cws, pos = pipe.analyze(text, word2id_c, id2tag_c, zy, word2id_p, id2tag_p, word2id_n, id2tag_n)
            pos = [pos[i][1] for i in range(len(pos))]
            if use_speech_tags_filter == True:
                cws = [cws[i] for i in range(len(pos)) if pos[i] in self.default_speech_tag_filter]
                pos = [pos[i] for i in range(len(pos)) if pos[i] in self.default_speech_tag_filter]

            word_list = [cws[i].strip() for i in range(len(pos)) if pos[i] !='x']
            word_list = [word for word in word_list if len(word)>0] 
        
        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list
        
    def segment_sentences(self, pic, pipe, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False, use_jieba = False ):
        """将列表sequences中的每个元素/句子转换为由单词构成的列表。
        
        sequences -- 列表，每个元素是一个句子（字符串类型）
        """
        
        res = []
        for sentence in sentences:
            res.append(self.segment(pic, pipe, text=sentence, 
                                    lower=lower, 
                                    use_stop_words=use_stop_words, 
                                    use_speech_tags_filter=use_speech_tags_filter, 
                                    use_jieba = use_jieba ))
        return res
        
class SentenceSegmentation(object):
    """ 分句 """
    
    def __init__(self, delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        self.delimiters = set([util.as_text(item) for item in delimiters])
    
    def segment(self, text):
        res = [util.as_text(text)]
        
        util.debug(res)
        util.debug(self.delimiters)

        for sep in self.delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res 
        
class Segmentation(object):
    
    def __init__(self, stop_words_file = None, 
                    allow_speech_tags = util.allow_speech_tags,
                    delimiters = util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimiters)
        
    def segment(self, pic, pipe, text, lower = False, use_jieba = False):
        text = util.as_text(text)
        sentences = self.ss.segment(text) # Sentences is a sentence with end delimiters .
        words_no_filter = self.ws.segment_sentences(pic, pipe, sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = False,
                                                    use_speech_tags_filter = False,
                                                    use_jieba = use_jieba )
        words_no_stop_words = self.ws.segment_sentences(pic, pipe, sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = True,
                                                    use_speech_tags_filter = False,
                                                    use_jieba = use_jieba )

        words_all_filters = self.ws.segment_sentences(pic, pipe, sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = True,
                                                    use_speech_tags_filter = True,
                                                    use_jieba = use_jieba )

        return util.AttrDict(
                    sentences           = sentences, 
                    words_no_filter     = words_no_filter, 
                    words_no_stop_words = words_no_stop_words, 
                    words_all_filters   = words_all_filters
                )
    
        

if __name__ == '__main__':
    pass
