# author: sunshine
# datetime:2021/7/29 下午4:53
from argparse import Namespace
from transformers import BertTokenizerFast
from src.re.muti_head.data_loader import load_data, MutiHeadDataset, load_mapping
from src.re.muti_head.train import Trainer
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
        data_path='dataset/re',
        output='models',
        bert_path='/home/sunshine/pre_models/pytorch/bert-base-chinese/',
        train_mode='train',
        head_type='Biaffine'
    )
    return Namespace(**params)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(args, tokenizer, debug=False):
    """生成数据迭代器
    """

    train_data = load_data(args.data_path + '/train_data.json')
    valid_data = load_data(args.data_path + '/dev_data.json')

    if debug:
        train_data = train_data[:20]
        valid_data = valid_data[:10]
    p2id, e2id = load_mapping(args.data_path + '/all_50_schemas')

    train_data_loader = MutiHeadDataset(
        data=train_data,
        tokenizer=tokenizer,
        predict2id=p2id,
        entity2id=e2id,
        max_len=256
    ).get_data_loader(batch_size=args.batch_size)

    valid_data_loader = MutiHeadDataset(
        data=valid_data,
        tokenizer=tokenizer,
        predict2id=p2id,
        entity2id=e2id,
        max_len=256
    ).get_data_loader(batch_size=args.batch_size)

    return (train_data_loader, valid_data_loader), p2id, e2id


def main():
    args = get_args()
    # 设置随机种子
    set_seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
    # 处理数据
    data_loader, p2id, e2id = build_dataset(args, tokenizer)

    # 构建trainer

    trainer = Trainer(
        args=args,
        data_loaders=data_loader,
        tokenizer=tokenizer,
        id2p={v: k for k, v in p2id.items()},
        id2e={v: k for k, v in e2id.items()}
    )

    trainer.train()


if __name__ == '__main__':
    main()
