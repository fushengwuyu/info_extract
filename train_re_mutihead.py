# author: sunshine
# datetime:2021/7/23 上午10:17
import torch
from transformers import BertTokenizerFast
from src.spo_mutihead import SPOBiaffine
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from functools import partial
import numpy as np
import json
import random
from argparse import Namespace


class MutiHeadDataset(Dataset):
    def __init__(self, data_path, tokenizer, predict2id, entity2id, max_len=256, is_train=True):
        super().__init__()
        self.data = self.load_data(data_path)[:1000]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.entity2id = entity2id
        self.predict2id = predict2id
        self.is_train = is_train

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def create_collate_fn(self):
        def find_index(index, patten):
            for i in range(len(index)):
                if index[i:i + len(patten)] == patten:
                    return (i, i + len(patten) - 1)
            return -1

        def collate(examples):
            batch_start_tokens, batch_end_tokens, batch_so, batch_relations = [], [], [], []
            texts = [t[0] for t in examples]
            inputs = self.tokenizer(texts, padding='longest', max_length=self.max_len,
                                    truncation='longest_first', return_offsets_mapping=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']

            batch_input_ids = torch.tensor(input_ids, dtype=torch.long)
            batch_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            batch_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

            batch_gold_answers = []
            if not self.is_train:
                """验证
                """
                for example in examples:
                    spo_list = example[1]
                    gold = []
                    for spo in spo_list:
                        gold.append((spo['subject'], spo['predicate'], spo['object']))
                    batch_gold_answers.append(gold)
                return [batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_gold_answers, texts,
                        inputs['offset_mapping']]
            else:
                """训练
                """
                pading_len = len(input_ids[0])
                for example, ids in zip(examples, input_ids):
                    text, spo_list = example
                    start_tokens = np.zeros((pading_len, len(self.entity2id)))
                    end_tokens = np.zeros((pading_len, len(self.entity2id)))
                    so = np.zeros((pading_len,))
                    relation = np.zeros((pading_len, pading_len, len(self.predict2id)))
                    for spo in spo_list:
                        s_typeid = self.entity2id[spo['subject_type']]
                        s_token = self.tokenizer(spo['subject'])['input_ids'][1:-1]

                        o_typeid = self.entity2id[spo['object_type']]
                        o_token = self.tokenizer(spo['object'])['input_ids'][1:-1]

                        s_index = find_index(ids, s_token)
                        o_index = find_index(ids, o_token)

                        if s_index != -1 and o_index != -1:
                            start_tokens[s_index[0], s_typeid] = 1
                            end_tokens[s_index[1], s_typeid] = 1

                            start_tokens[o_index[0], o_typeid] = 1
                            end_tokens[o_index[1], o_typeid] = 1

                            so[s_index[1]] = s_typeid
                            so[o_index[1]] = o_typeid

                            relation[s_index[1], o_index[1], self.predict2id[spo['predicate']]] = 1

                    batch_start_tokens.append(start_tokens)
                    batch_end_tokens.append(end_tokens)
                    batch_so.append(so)
                    batch_relations.append(relation)

                batch_start_tokens = torch.tensor(batch_start_tokens, dtype=torch.long)
                batch_end_tokens = torch.tensor(batch_end_tokens, dtype=torch.long)
                batch_so = torch.tensor(batch_so, dtype=torch.long)
                batch_relations = torch.tensor(batch_relations, dtype=torch.long)

                return [batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_start_tokens,
                        batch_end_tokens, batch_so, batch_relations]

        return partial(collate)

    def load_data(self, path):

        D = []
        with open(path, 'r', encoding='utf-8') as rd:
            for line in rd:
                d = json.loads(line)
                D.append([d['text'], d['spo_list']])

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

        self.predicate2id, self.id2predicate, entity = self.load_mapping(cfg.corpus_path + '/all_50_schemas')
        self.ent2id = {e: i + 1 for i, e in enumerate(entity)}
        self.ent2id.update({"O": 0})
        self.model = SPOBiaffine.from_pretrained(cfg.bert_path, p_num=len(self.predicate2id), e_num=len(self.ent2id),
                                                 max_len=256)

        self.model.to(self.device)
        self.train_loader, self.dev_loader = self.setup()
        self.optimizer, self.schedule = self.configure_optimizers_adam()

    def load_mapping(self, schema_path):
        predicate2id, id2predicate, entity = {}, {}, set()
        with open(schema_path, 'r', encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                if l['predicate'] not in predicate2id:
                    id2predicate[len(predicate2id)] = l['predicate']
                    predicate2id[l['predicate']] = len(predicate2id)
                entity.add(l["object_type"])
                entity.add(l['subject_type'])
        return predicate2id, id2predicate, entity

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

            # result = self.evaluate()
            # print(result)
            # if result['f1'] > best_score:
            #     best_score = result['f1']
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
        input_ids, attention_mask, token_type_ids, start_labels, end_labels, so, p_labels = batch
        loss = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, cs_ids=so,
                          labels=[start_labels, end_labels, p_labels])
        return loss

    def validation_step(self, batch):
        A, B, C = 1e-12, 1e-12, 1e-12
        gold_answers, texts, mapping = batch[3:]
        batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch[:3])
        input_ids, attention_mask, token_type_ids = batch
        batch_subject, batch_end_list, out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out_logits = out.cpu().numpy()

        for entities, end_list, out, m, text, gold in zip(batch_subject, batch_end_list, out_logits, mapping, texts, gold_answers):
            answer = set()
            s_end, o_end, predict = np.where(out > 0)
            for _s, _o, p in zip(s_end, o_end, predict):
                if _s in end_list and _o in end_list:
                    sub = entities[end_list.index(_s)]
                    sub_t = text[m[sub[0]][0]: m[sub[-1]][-1]]

                    obj = entities[end_list.index(_o)]
                    obj_t = text[m[obj[0]][0]: m[obj[-1]][-1]]
                    answer.add((sub_t, self.id2predicate[p], obj_t))

            gold = set(gold)
            a = len(gold & answer)
            b = len(answer)
            c = len(gold)
            A += a
            B += b
            C += c

        return A, B, C

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

        train_dataset = MutiHeadDataset(self.cfg.corpus_path + '/train_data.json', self.tokenizer, self.predicate2id,
                                        self.ent2id, max_len=256, is_train=True)
        dev_dataset = MutiHeadDataset(self.cfg.corpus_path + '/dev_data.json', self.tokenizer, self.predicate2id,
                                      self.ent2id, max_len=256, is_train=False)
        # 若并行加载数据，不可用于调试
        if self.debug:
            num_workers = 0
        else:
            num_workers = 2
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=train_dataset.create_collate_fn(), num_workers=num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                                collate_fn=dev_dataset.create_collate_fn(), num_workers=num_workers)
        return train_loader, dev_loader


if __name__ == '__main__':
    cfg = {
        "bert_path": "/home/sunshine/pre_models/pytorch/bert-base-chinese",
        "corpus_path": "/home/sunshine/python/Info_extract/dataset/re",
        "learning_rate": 2e-5,
        "batch_size": 4,
        "seed": 2333,
        "epoch_num": 2,
        'output': ".",
        "accum_iter": 1,
        "max_norm": 1,
        "num_class": 9,
        'rewarm_epoch_num': 2,
        'T_mult': 1

    }
    args = Namespace(**cfg)
    trainer = Trainer(args, use_gpu=False, debug=True)
    # trainer.fit()
    trainer.resume('models/pytorch_SPOBiaffine.bin')

    result = trainer.evaluate()
    print(result)