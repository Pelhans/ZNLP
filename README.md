# ZNLP
Chinese NLP package

ZLP 是一个完全基于神经网络的自然语言处理工具包，框架采用谷歌的tensorflow，内容涵盖中文分词/ 词性标注 / 
命名实体识别 / 依存句法分析 / 关键词提取 / 文本摘要 / 文本分类 。

## 内容列表
-安装<br>
-模块<br>
-使用<br>
-其他<br>

## 安装
直接从github上clone。即  git clone https://github.com/Pelhans/ZNLP   
运行./requirement.sh 来安装依赖    
### 依赖
-tensorflow>= 1.3(更低版本的没试过，不过高于1.0的应该就可以)<br>
-pandas<br>
-numpy<br>
-tqdm<br>
sklearn<br>
scipy<br>
networkx(textrank需要)<br>
jieba(textrank可选)<br>
以上几项均通过pip 安装或运行./requirement.sh，目前仅支持python2<br>

## 模块
包含以下模块：<br>
--中文分词: 基于BLSTM神经网络，采用微软bakekoff分词语料。<br>
--词性标注： 基于BLSTM神经网络，采用人民日报1998年上半年词性标注语料库。<br>
--命名实体识别： 基于BLSTM神经网络，采用由1998年人民日报词性标注语料库处理得到的命名实体识别语料库。<br>
--依存句法分析：采用Danqi 和Mannning 2014年论文提出的神经网络结构，语料库采用清华大学语义依存分析语料库。<br>
--关键词提取：基于TextRank算法，分词&词性标注采用库内模块，也可使用结巴分词。。。<br>
--文本摘要：同样基于TextRank算法，与关键词提取处于同一模块。<br>


## 使用
单个模块的训练请参考对应文件夹下面的README.md，下面对各个模块的调用做一个说明：    
### 分词模块

from znlp.loadModel import LoadModel    
from znlp.loadModel import show_cws, show_pos    
sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样。'    
cws = LoadModel('cws')    
result = cws.predict(sentence)    
show_cws(result)    

结果：    
人们 思考 问题 往往 不是 从 零 开始 的 。 就 像 你 好现在 阅读 这 篇 文章 一样 。
### 词性标注模块
from znlp.loadModel import LoadModel    
from znlp.loadModel import show_cws, show_pos    
    
sentence = u'词法 分析 终于 完成 了 。 这 都 叫 啥 事 啊 。'    
pos = LoadModel('pos')     
result = pos.predict(sentence)    
show_pos(result)    

结果：    
词法/l 分析/v 终于/d 完成/v 了/y 。/w 这/r 都/d 叫/v 啥/r 事/n 啊/y 。/w

### 命名实体识别
from znlp.loadModel import LoadModel    
from znlp.loadModel import show_cws, show_pos    
    
sentence = u'我 爱 吃 北京 烤鸭 。'    
ner = LoadModel('ner')    
result = ner.predict(sentence)    
show_pos(result)    

结果：    
我/nan 爱/nan 吃/nan 北京/nt 烤鸭/nan 。/nan

### 依存句法分析
from znlp import dparser

word = [u"世界", u"第", u"八", u"大", u"奇迹", u"出现"]    
pos = [u"n",u"m",u"m",u"a",u"n",u"v"]    
parser = dparser.ParserLoader( word, pos)    
UAS, LAS, token_num, token_dep =  parser.predict(parser.model, parser.dataset)    
parser.print_conll(token_num, token_dep)    

## 其他
如有问题，可以发邮件联系我:me@pelhans.com
