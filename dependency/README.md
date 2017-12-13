# Neural_dependency_parsing

## 项目介绍 
基于[Chen and Manning](https://cs.stanford.edu/%7Edanqi/papers/emnlp2014.pdf) 论文的一个神经网络依存句法分析工具。框架使用tensorflow。<br>
项目参考自 [akjindal53244/dependency_parsing_tf](https://github.com/akjindal53244/dependency_parsing_tf) 。感谢原作者<br>
原项目是用于英文的，这里将其精简并修改为中文的分析器。<br>
<br>
## 依赖
tensorflow >=1.3, #更低的应该也可以<br>
<br>
## 测试结果
现在为初始版本，使用THU的UAS结果为 78.299056073，LAS只有5.2....。后续有时间会对其进行更新<br>
测试输出：<br>
1 世界 世界 n n _ 5 连接依存<br>
2 第 第 m m _ 5 描述<br>
3 八 八 m m _ 2 数量<br>
4 大 大 a a _ 5 限定<br>
5 奇迹 奇迹 n n _ 6 经验者<br>
6 出现 出现 v v _ 0 核心成分<br>
<br>
Gold原文：<br>
1   世界    世界    n   n   _   5   限定    <br>
2   第  第  m   m   _   4   限定    <br>
3   八  八  m   m   _   2   连接依存    <br>
4   大  大  a   a   _   5   限定    <br>
5   奇迹    奇迹    n   n   _   6   存现体  <br>
6   出现    出现    v   v   _   0   核心成分<br>
可以看出依赖弧标记还是可以的，但依存关系标签的话。。。等我有时间一定解决掉。<br>
<br>
## 运行
python parser_model.py<br>
## 其他
如有问题，可以发邮件联系我:p.zhang1992@gmail.com
