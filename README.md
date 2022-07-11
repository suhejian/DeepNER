# DeepNER

实现NER经典模型，仅用于个人学习

## 环境
- Linux系统
- GPU型号：GeForce RTX 3090
- python: 3.8
- pytorch: `torch==1.10.1+cu111` 
- 其它：`requirements.txt`文件中有相关依赖包

## 使用
1. 安装虚拟环境`conda create -n deepner python=3.8`
2. 激活虚拟环境`conda activate deepner`
3. 安装PyTorch：`pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
    可根据自己的GPU型号，CUDA版本等自行选择合适的PyTorch版本
4. 安装其它依赖包：`pip install -r requirements.txt`
5. 运行：`CUDA_VISIBLE_DEVICES=0 python run.py`

## 待完成
1. 数据统计
 - [ ] 对数据集整体的分析
2. 数据格式转换
 - [x] CLUENER数据格式转换成CONLL格式
2. 模型
 - [x] BERT
 - [ ] BERT+CRF