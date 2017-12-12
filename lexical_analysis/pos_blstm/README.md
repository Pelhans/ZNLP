# ner_blstm
NER for PFR corpus

采用1998年人民日报词性标注语料库，模型为BLSTM神经网络模型

系统要求:<br>
tensorflow>= 1.3(更低版本的没试过，不过高于1.0的应该就可以)<br>
pandas<br>
numpy<br>
tqdm<br>

运行：<br>
<br>
download_data.py 用来下载语料库<br>

gen_ner_file.py<br>
    1.将POS语料库中的[国务院/ns 侨办/ns]nt 转化为 国务院侨办/nt<br>
    2.将除nz nt nr ns 外的词性标记转换为nan。<br>

get_pickle.py 用来将词和标签转换为id并存为pkl。_vaild 和 _test 文件对应于生成验证和测试数据<br>
ner_blstm.py用来训练模型<br>
ner_test.py用来测试<br>

<br>
测试结果（包含非实体词的标记准确率）:<br>
Epoch training 15678, acc=0.995077, cost=0.0154902, speed=6.23596 s/epoch<br>
TEST RESULT:<br>
Test 1960, acc=0.97513, cost=0.10224<br>


输入句子测试：我/nan 爱/nan 吃/nan 北京/ns 烤鸭/nan 。/nan <br>


如有问题，可以发邮件联系我:p.zhang1992@gmail.com<br>
