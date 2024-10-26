# hanlp-train
基于hanlp训练中文分词和词性标注混合模型。

## 使用

### 环境安装

Python 3.10

```python
Pytorch        2.0.1  # 需要支持GPU的版本
transformers   4.33.2
hanlp          2.1.0b50
```


### 数据集

需要准备分词和词性标注的数据集。

目录结构如下

```shell
-- dataset
   |-- cws  # 分词
      |- tain.txt
      |- dev.txt
      |- test.txt
   |-- pos  # 词性标注
      |- train.tsv
      |- dev.tsv
      |- test.tsv
```

样例如 `data/sample` 文件夹，如果数据不够多，验证集dev内容可以和test一样。

> 由于是直接基于预训练bert模型训练的，因此建议将自行构建的数据集切分后与 [CTB8数据集](https://bbs.hankcs.com/t/topic/27)合并后再进行训练。

### 训练

训练分词和词性标注混合模型代码如 `train_hanlp_tok_pos.py`

```python
from train_hanlp_tok_pos import train, test

if __name__=="__main__":
    # 需要下载预训练模型 hfl/chinese-electra-180g-base-discriminator
    pretrain_bert_model = "huggingface/hfl-chinese-electra-180g-base-discriminator"  # 预训练模型
    data_path = "data/sample/"  # 数据集
    model_save_path = 'model/hanlp_tok_pos_electra_base'  # 模型保存地址
  
    # 训练 
    train(pretrain_bert_model, data_path, model_save_path, epoch=5, batch_size=32)
    # 测试
    test(model_save_path)  # 包含3个案例，如果要了解使用方法，看具体test函数
```

训练需要**下载** `hfl/chinese-electra-180g-base-discriminator` 预训练模型，推荐去[huggingface](https://huggingface.co/hfl/chinese-electra-180g-base-discriminator)或[modelscope](https://modelscope.cn/models/dienstag/chinese-electra-180g-base-discriminator) 下载。

> 同时训练分词和词性标注，当批次大小=`32`，大约需要 `11GB`显存。

除了`hfl/chinese-electra-180g-base-discriminator`，大部分的中文Bert模型都可以，如

| 模型                                         | huggingface                                                  | modelscope                                                   |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| hfl/chinese-electra-180g-base-discriminator  | [地址](https://huggingface.co/hfl/chinese-electra-180g-base-discriminator) | [地址](https://modelscope.cn/models/dienstag/chinese-electra-180g-base-discriminator) |
| hfl/chinese-electra-180g-small-discriminator | [地址](https://huggingface.co/hfl/chinese-electra-180g-small-discriminator) | [地址](https://modelscope.cn/models/dienstag/chinese-electra-180g-small-discriminator) |
| hfl/chinese-macbert-base                     | [地址](https://huggingface.co/hfl/chinese-macbert-base)      | [地址](https://modelscope.cn/models/dienstag/chinese-macbert-base) |
| hfl/chinese-roberta-wwm-ext                  | [地址](https://huggingface.co/hfl/chinese-roberta-wwm-ext)   | [地址](https://modelscope.cn/models/dienstag/chinese-roberta-wwm-ext) |





## 自定义数据集

样例如 `data/sample` 文件夹

**分词数据样例**

> 每个词使用空格隔开，一行一个样本句子

```shell
经营 大棚 蔬菜 比较 辛苦 。
看 得 出来 ， 她 很 开心 。
```

**词性标注数据样例**

> 每一行包括一个词及其词性(通过`\t` 隔开)，样本之间通过一个**空白行分割**

```
经营	VV
大棚	NN
蔬菜	NN
比较	AD
辛苦	VA
。	PU

看	VV
得	DER
出来	VV
，	PU
她	PN
很	AD
开心	VA
。	PU
```

> 词性说明 [ctb — HanLP Documentation (hankcs.com)](https://hanlp.hankcs.com/docs/annotations/pos/ctb.html)

## CTB8测试结果

| 预训练模型                               | 分词效果（F1） | 词性标注效果(ACC) |
| ---------------------------------------- | -------------- | ----------------- |
| 官方混合模型                             | **97.74%**     | 96.57%            |
| chinese-electra-180g-base-discriminator  | **97.58%**     | **97.14%**        |
| chinese-electra-180g-small-discriminator | 97.15%         | 96.52%            |
| chinese-macbert-base                     | 97.44%         | 97.11%            |
| chinese-roberta-wwm-ext                  | 97.43%         | 97.08%            |

> 官方模型结果仅供参考，因为官方使用的数据还包含除ctb8其他数据，因此实际结果应该更好一些

经过测试，除了 `chinese-electra-180g-small-discriminator` 模型比较小，效果最差，同等级的BERT模型效果相差不大。


## 感谢

[hankcs/HanLP](https://github.com/hankcs/HanLP)
