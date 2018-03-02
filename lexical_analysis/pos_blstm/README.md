# pos_blstm
NER for PFR corpus

采用1998年人民日报词性标注语料库，模型为BLSTM神经网络模型


运行：<br>
<br>

utils/get_pickle.py 用来将词和标签转换为id并存为pkl。可附加参数 dev 或 test, 对应于生成验证和测试数据<br>
bin/pos_blstm.py用来训练模型<br>
bin/pos_test.py用来测试<br>

<br>
测试结果:<br>
Epoch training 22564, acc=0.975644, cost=0.081708, speed=25.8058 s/epoch<br>
TEST RESULT:<br>
Test 1960, acc=0.98525, cost=0.0491332<br>

输入句子测试：
POS标记结果为: 
词法/j 分析/v 终于/d 完成/v 了/y 。/w 这/r 都/d 叫/v 啥/r 事/n 啊/y 。/w 
-0.606605  s

如有问题，可以发邮件联系我:me@pelhans.com<br>
