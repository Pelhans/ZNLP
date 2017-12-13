# Readme
<br>
本repo是为了将cws pos 和ner的pipeline实现。即输入一句话后得到该句子的分词、词性标注、命名实体识别结果。<br>
<br>
本repo依赖于cws & pos & ner repo,先跟据这三部分的Readme训练好模型而后调用p运行pipeline或根据pipeline例子调用相应接口<br>
<br>
最终运行python pipeline.py 来实现三个功能的pipeline。<br>
<br>
测试结果：<br>
<br>
请您输入一句话：天津机场的服务特别好。<br>
您输入的句子为： 天津机场的服务特别好。<br>
分词结果为： 天津 机场 的 服务 特别 好 。<br>
词性标注结果为： 天津/ns 机场/n 的/u 服务/vn 特别/d 好/a 。/w<br>
命名实体识别结果为： 天津/ns 机场/nan 的/nan 服务/nan 特别/nan 好/nan 。/nan<br>
<br>
