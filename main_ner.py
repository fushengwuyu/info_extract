# author: sunshine
# datetime:2021/7/23 下午4:12

from argparse import Namespace
from transformers import BertTokenizerFast
from src.ner.data_loader import load_data, NERDataset
from src.ner.train import Trainer
import numpy as np
import random
import torch


def get_args():
    params = dict(
        max_len=128,
        batch_size=4,
        drop=0.3,
        epoch_num=10,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        seed=2333,
        data_path='dataset/ner/',
        output='models',
        bert_path='/home/sunshine/pre_models/pytorch/bert-base-chinese/',
        train_mode='train',
        head_type='Biaffine'  # [GlobalPointer, Biaffine, Mutihead, TxMutihead]
    )
    return Namespace(**params)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(args, tokenizer):
    """生成数据迭代器
    """
    train_data, l1 = load_data(args.data_path + '/train_data.txt')
    valid_data, l2 = load_data(args.data_path + '/val_data.txt')

    l1.update(l2)
    label2id = {v: k for k, v in enumerate(l1)}

    train_data_loader = NERDataset(
        data=train_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=256
    ).get_data_loader(batch_size=args.batch_size)

    valid_data_loader = NERDataset(
        data=valid_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=256
    ).get_data_loader(batch_size=args.batch_size)

    return (train_data_loader, valid_data_loader), label2id


def main():
    args = get_args()
    # 设置随机种子
    set_seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
    # 处理数据
    data_loader, label2id = build_dataset(args, tokenizer)
    print(len(label2id))
    # 构建trainer

    trainer = Trainer(
        args=args,
        data_loaders=data_loader,
        tokenizer=tokenizer,
        label2id=label2id
    )

    trainer.train(args)


if __name__ == '__main__':
    main()
