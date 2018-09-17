基于CRF++的词性标注，只需要更换输入文件就可以用在其他序列标注任务中。

* 运行install_crf++.sh 安装crf++工具包    
* 运行transfer_format.py 将人民日报词性标注语料库转换为 CRF++指定的格式。    
* 修改 template 文件设置自己的模板，我这里 Unigram 和 Bigram都用了    
* crf_learn -f 3 -p 8 -c 4.0 template train.data model 训练模型    
* 根据训练得到的model 来对测试数据集进行词性标注 crf_test -m model test.data > test.rst    
* 通过命令 python calc_p.py test.rst 计算准确率。

您可以在[坚果云上](https://www.jianguoyun.com/p/DdjH8K0Qq_6CBxi7nnM)下载模型和对应数据集到当前目录，直接使用。
