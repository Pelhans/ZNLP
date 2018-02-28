@author pelhans<br>
data:14/11/2017<br>


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

测试语句：<br>
total_sen_num:  9<br>
人们 / 思考 / 问题 / 往往 / 不是 / 从 / 零 / 开始 / 的 / 。 / 就 / 好 / 像 / 你 / 现在 / 阅读 / 这 / 篇 / 文章 / 一样 / ， / 你 / 对 / 每个 / 词 / 的 / 理解 / 都会 / 依赖 / 于 / 你 / 前面 / 看到 / 的 / 一些 / 词， /    /    /    / 而 / 不是 / 把 / 你 / 前面 / 看 / 的 / 内容 / 全部 / 抛弃 / 了 / ， / 忘记 / 了 / ， / 再去 / 理解 / 这个 / 单词 / 。 / 也 / 就 / 是 / 说， / 人们 / 的 / 思维 / 总是 / 会 / 有 / 延续 / 性 / 的 / 。 / <br>
0.398095 s<br>
输入句子越长优势越明显。<br>
