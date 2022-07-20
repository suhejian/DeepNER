# DeepNER

实现NER经典模型，仅用于个人学习

## 环境

我个人的开发环境是：
- Linux系统
- GPU型号：GeForce RTX 3090
- python: 3.8
- pytorch: `torch==1.10.1+cu111` 
- 其它：`requirements.txt`文件中有相关依赖包

需要注意的是，PyTorch版本过低的话在GeForce RTX 3090上会出现不兼容问题。如果你的GPU型号不是3090，你可以根据实际情况选择合适的PyTorch版本。

## 一些说明

暂时只针对中文的NER数据

### 数据格式

NER最常用的范式就是**序列标注**，输入是一个`token`序列，输出是一个标签序列。虽然`conll`格式是最常用的，但是我这里是用JSON格式存储每个样本的。
以下是一个样例：

``` json
{"text": "生生不息CSOL生化狂潮让你填弹狂扫", "labels": ["O", "O", "O", "O", "B-game", "I-game", "I-game", "I-game", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
```

由于我用的数据集都是中文数据集，将`text`的内容以字符为单位分隔即可得到`token`序列，`labels`的内容即为标签序列。

## 使用

1. 下载`git clone git@github.com:suhejian/DeepNER.git`
2. 进入项目所在目录：`cd DeepNER`
3. 安装虚拟环境`conda create -n deepner python=3.8`
4. 激活虚拟环境`conda activate deepner`
5. 安装PyTorch：`pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
    可根据自己的GPU型号，CUDA版本等自行选择合适的PyTorch版本
6. 安装其它依赖包：`pip install -r requirements.txt`
7. 运行：`CUDA_VISIBLE_DEVICES=0 python run.py`，模型默认是BERT, 使用提供的CLUENER数据集。

可以将自己的数据集格式转换一下，改变相应的命令行参数

## 待完成
1. 数据统计
 - [ ] 对数据集整体的分析
2. 数据格式转换
 - [x] CLUENER数据格式转换成序列标注格式
2. 模型
 - [x] BERT
 - [ ] BERT+CRF


【参考】：
1. https://github.com/ljynlp/W2NER
2. https://github.com/lonePatient/BERT-NER-Pytorch
3. https://github.com/taishan1994/pytorch_GlobalPointer_triple_extraction