# ner_blstm
NER for PFR corpus

采用1998年人民日报词性标注语料库，模型为BLSTM神经网络模型

运行：<br>
<br>
download_data.py 用来下载语料库<br>

gen_ner_file.py<br>
    1.将POS语料库中的[国务院/ns 侨办/ns]nt 转化为 国务院侨办/nt<br>
    2.将除nz nt nr ns 外的词性标记转换为nan。<br>

utils/get_pickle.py 用来将词和标签转换为id并存为pkl。参数 dev 和 test 文件对应于生成验证和测试数据<br>
bin/ner_blstm.py用来训练模型<br>
bin/ner_test.py用来测试<br>

<br>
测试结果（包含非实体词的标记准确率）:<br>
Epoch training 15678, acc=0.996689, cost=0.0104492, speed=19.0691 s/epoch<br>
TEST RESULT:<br>
Test 1960, acc=0.988052, cost=0.0448491<br>


输入句子测试：我/nan 爱/nan 吃/nan 北京/ns 烤鸭/nan 。/nan <br>


如有问题，可以发邮件联系我:p.zhang1992@gmail.com<br>
