# -*- coding:utf-8 -*-
import os
from hanlp.common.dataset import SortingSamplerBuilder
from hanlp.common.transform import NormalizeCharacter
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp.components.mtl.tasks.pos import TransformerTagging
from hanlp.components.mtl.tasks.tok.tag_tok import TaggingTokenization
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from hanlp.utils.lang.zh.char_table import HANLP_CHAR_TABLE_JSON
from hanlp.utils.log_util import cprint


def train(pretrain_bert_model, data_path, model_save_path, epoch=5, batch_size=32, lr=1e-3, max_seq_len=512):
    """联合训练分词和词性标注

    Args:
        pretrain_bert_model (str): 预训练模型基座路径, 可以使用常见的中文bert模型, 如chinese-electra-180g-base-discriminator
        data_path (str): 数据集路径
        model_save_path (str): 训练完成后模型保存地址
        epoch (int, optional): 训练批次. Defaults to 5.
        batch_size (int): 批次大小, 默认32.
        lr (int): 学习率, 默认 1e-3.
        max_seq_len (int, optional): 基座模型支持的文本长度, 可以小于等于它. Defaults to 512.
    """
    # 1 导入数据集
    CTB8_POS_TRAIN = os.path.join(data_path, "pos/train.tsv")
    CTB8_POS_DEV = os.path.join(data_path, "pos/dev.tsv")
    CTB8_POS_TEST = os.path.join(data_path, "pos/test.tsv")
    CTB8_CWS_TRAIN = os.path.join(data_path, "cws/train.txt")
    CTB8_CWS_DEV = os.path.join(data_path, "cws/dev.txt")
    CTB8_CWS_TEST = os.path.join(data_path, "cws/test.txt")

    # 2 模型任务设置，下面设置了 tok分词和词性标注pos任务
    tasks = {
        'tok': TaggingTokenization(
            CTB8_CWS_TRAIN,  # 'tasks/cws/train.txt'
            CTB8_CWS_DEV,
            CTB8_CWS_TEST,
            SortingSamplerBuilder(batch_size=batch_size),
            max_seq_len=max_seq_len-2,
            hard_constraint=True,
            char_level=True,
            tagging_scheme='BMES',
            lr=lr,
            transform=NormalizeCharacter(HANLP_CHAR_TABLE_JSON, 'token'),
        ),
        'pos': TransformerTagging(
            CTB8_POS_TRAIN,  # https://wakespace.lib.wfu.edu/bitstream/handle/10339/39379/LDC2013T21.tgz#data/ + 'tasks/pos/train.tsv'
            CTB8_POS_DEV,
            CTB8_POS_TEST,
            SortingSamplerBuilder(batch_size=batch_size),
            hard_constraint=True,
            max_seq_len=max_seq_len-2,
            char_level=True,
            dependencies='tok',
            lr=lr,
        )
    }

    # 3 创建模型
    mtl = MultiTaskLearning()
    # 4 模型训练
    mtl.fit(
        ContextualWordEmbedding(
            'token',
            pretrain_bert_model,  # 使用本地预训练模型
            average_subwords=True,
            max_sequence_length=max_seq_len,
            word_dropout=.1
        ),
        tasks,
        model_save_path,
        epoch,
        lr=lr,
        encoder_lr=5e-5,
        grad_norm=1,
        gradient_accumulation=2,
        eval_trn=False,
    )
    cprint(f'Model saved in [cyan]{model_save_path}[/cyan]')
    # 评估测试集
    mtl.evaluate(model_save_path)


def test(model_save_path):
    mtl = MultiTaskLearning()
    # 使用
    mtl.load(model_save_path)
    # 分词和词性标注
    print("样例结果：")
    res = mtl('华纳音乐旗下的新垣结衣在12月21日于日本武道馆举办歌手出道活动')
    print(f"分词和词性标注 = {res}\n")

    a = 'HanLP 为 生产 环境 带来 次世代 最 先进 的 多 语种 NLP 技术 。'  # 官网结果= 'NR P NN NN VV NN AD JJ DEG CD NN NN NN PU'
    b = '我 的 希望 是 希望 张晚霞 的 背影 被 晚霞 映红 。'   # 官网结果 = 'PN DEG NN VC VV NR DEG NN LB NN VV PU' 

    # 单独进行分词
    res = mtl(['华纳音乐旗下的新垣结衣在12月21日于日本武道馆举办歌手出道活动'],
            tasks='tok')
    print(f"单独进行分词 = {res}\n")

    # 单独进行词性标注
    res = mtl([a.split(' '), b.split(' ')],
            tasks='pos',skip_tasks='tok')
    print(f"单独进行词性标注 = {res}\n")


if __name__=="__main__":
    # 预训练模型
    pretrain_bert_model = "huggingface/hfl-chinese-electra-180g-base-discriminator"
    # 数据集路径
    data_path = "data/sample/"
    # 训练后保存路径
    model_save_path = 'model/hanlp_tok_pos_electra_base'
    
    # 训练 
    train(pretrain_bert_model, data_path, model_save_path, epoch=5, batch_size=32)
    # 测试使用
    test(model_save_path)










