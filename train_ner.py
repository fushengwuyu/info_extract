# author: sunshine
# datetime:2021/7/23 上午10:17
import torch
from transformers import BertTokenizerFast
from src.ner_model import Net
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from functools import partial
import numpy as np
import json
import random
from argparse import Namespace


class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, ent2id=None, max_len=256):
        super().__init__()
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_special_tokens = True
        self.ent2id = ent2id

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def create_collate_fn(self):

        def find_index(offset_mapping, index):
            for i, offset in enumerate(offset_mapping):
                if offset[0] <= index <= offset[1]:
                    return i
            return -1

        def collate(examples):
            texts = [e['text'] for e in examples]
            inputs = self.tokenizer(texts, padding='longest', max_length=self.max_len, truncation='longest_first',
                                    return_offsets_mapping=True)
            token_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

            offset_mapping = inputs['offset_mapping']

            labels = [e['entity_list'] for e in examples]
            label_span = torch.zeros((len(texts), len(self.ent2id), len(offset_mapping[0]), len(offset_mapping[0])),
                                     dtype=torch.long)
            for i in range(len(texts)):

                for l in labels[i]:
                    """
                    labels: (3, 7, pro)
                    offset_mapping: [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 0)]
                    """
                    s = find_index(offset_mapping[i], l[0])
                    e = find_index(offset_mapping[i], l[1])

                    label_span[i, self.ent2id[l[2]], s, e] = 1

            return token_ids, attention_mask, token_type_ids, label_span

        return partial(collate)

    def load_data(self, path):
        """加载数据
        """
        D = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                D.append(item)
        return D


class Trainer:

    def __init__(self, cfg, use_gpu=True, debug=False):

        self.cfg = cfg
        self.debug = debug
        self.set_seed()
        if use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_path, do_lower_cast=True)

        self.ent2id = json.load(open(cfg.corpus_path + '/ent2id.json', 'r', encoding='utf-8'))
        self.model = Net.from_pretrained(cfg.bert_path, head_type='EfficientGlobalPointer', num_class=len(self.ent2id),
                                         max_len=256)
        self.model.to(self.device)
        self.train_loader, self.dev_loader = self.setup()
        self.optimizer, self.schedule = self.configure_optimizers_adam()


    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def configure_optimizers_adam(self):

        T_mult = 1
        rewarm_epoch_num = 2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(self.train_loader) * rewarm_epoch_num,
                                                                         T_mult)
        return optimizer, scheduler

    def fit(self):

        self.model.train()
        step_gap = 10
        best_score = 1e-10
        for epoch in range(int(self.cfg.epoch_num)):

            gap_loss = 0.0
            for step, batch in enumerate(self.train_loader, 1):

                loss = self.training_step(batch)
                loss = loss / self.cfg.accum_iter
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)

                if (step % self.cfg.accum_iter == 0) or (step == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.schedule.step()

                gap_loss += loss
                if step % step_gap == 0:
                    current_loss = gap_loss / step_gap
                    msg = "step {} / {} of epoch {}, train/loss: {}".format(step, len(self.train_loader),
                                                                            epoch, current_loss)
                    print(msg)
                    gap_loss = 0.0

            result = self.evaluate()
            print(result)
            if result['f1'] > best_score:
                best_score = result['f1']
                self.save(self.cfg.output + '/pytorch_{}.bin'.format(self.model.head.__class__.__name__))

    def save(self, output_model_file):
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), str(output_model_file))

    def resume(self, resume_model_file):
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def training_step(self, batch):

        batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch)
        input_ids, attention_mask, token_type_ids, label_span = batch
        # batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch
        loss = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label_span)

        # loss = self.criterion(label_span, logits)
        return loss

    def validation_step(self, batch):
        batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch)
        input_ids, attention_mask, token_type_ids, label = batch
        y_pred = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        y_pred = torch.greater(y_pred, 0)
        temp_A = torch.sum(label * y_pred).item()
        temp_B = torch.sum(label).item()
        temp_C = torch.sum(y_pred).item()
        return temp_A, temp_B, temp_C

    @torch.no_grad()
    def evaluate(self):
        """验证
        """
        self.model.eval()
        A, B, C = 1e-10, 1e-10, 1e-10

        for batch in tqdm(self.dev_loader):
            a, b, c = self.validation_step(batch=batch)
            A += a
            B += b
            C += c
        self.model.train()

        return {"f1": 2 * A / (B + C), 'p': A / B, 'r': A / C}

    def setup(self):
        """
        准备数据集
        :return:
        """

        train_dataset = NERDataset(self.cfg.corpus_path + '/train.json', self.tokenizer, ent2id=self.ent2id,
                                   max_len=256)
        dev_dataset = NERDataset(self.cfg.corpus_path + '/dev.json', self.tokenizer, ent2id=self.ent2id,
                                 max_len=256)
        # 若并行加载数据，不可用于调试
        if self.debug:
            num_workers = 0
        else:
            num_workers = 12
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=train_dataset.create_collate_fn(), num_workers=num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                                collate_fn=train_dataset.create_collate_fn(), num_workers=num_workers)
        return train_loader, dev_loader


if __name__ == '__main__':
    cfg = {
        "bert_path": "/home/sunshine/pre_models/pytorch/bert-base-chinese",
        "corpus_path": "/home/sunshine/python/Info_extract/dataset/ner",
        "learning_rate": 5e-5,
        "batch_size": 2,
        "seed": 2333,
        "epoch_num": 10,
        'output': ".",
        "accum_iter": 1,
        "max_norm": 1,
        "num_class": 9,
        'rewarm_epoch_num': 2,
        'T_mult': 1

    }
    args = Namespace(**cfg)
    trainer = Trainer(args, use_gpu=False, debug=True)
    trainer.fit()
