@author pelhans<br>
data:14/11/2017<br>

参考 https://github.com/yongyehuang/Tensorflow-Tutorial/blob/master/Tutorial_6%20-%20Bi-directional%20LSTM%20for%20sequence%20labeling%20(Chinese%20segmentation).ipynb 进行修改，整合后完成

系统要求:<br>
tensorflow>= 1.3(更低版本的没试过，不过高于1.0的应该就可以)<br>
pandas<br>
numpy<br>
tqdm<br>

运行：<br>
<br>
运行 python get_pickle.py  来得到训练数据<br>
运行 python cws_blstm.py 来训练你的模型<br>
运行 python cws_test.py 来通过输入句子测试模型<br>
<br>
测试结果:<br>
Epoch training 205968<br>
acc=0.938678, cost=0.166657, speed=124.181 s/epoch<br>
<br>
TEST RESULT:<br>
Test 64366, acc=0.942572, cost=0.155533<br>

如有问题，可以发邮件联系我:p.zhang1992@gmail.com<br>
