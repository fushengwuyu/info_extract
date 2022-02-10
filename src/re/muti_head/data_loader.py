# author: sunshine
# datetime:2021/7/28 下午5:34
from torch.utils.data import DataLoader, Dataset
import json
from functools import partial
import torch
import numpy as np


def load_data(path):
    D = []
    with open(path, 'r', encoding='utf-8') as rd:
        for line in rd:
            d = json.loads(line)
            D.append([d['text'], d['spo_list']])

    return D


def load_mapping(path):
    """
    {"object_type": "地点", "predicate": "祖籍", "subject_type": "人物"}
    """
    predict, entity = set(), set()
    with open(path, 'r', encoding='utf-8') as rd:
        for line in rd:
            d = json.loads(line)
            predict.add(d['predicate'])
            entity.add(d["object_type"])
            entity.add(d['subject_type'])
    predict2id = {v: k for k, v in enumerate(predict)}
    entity2id = {v: k + 1 for k, v in enumerate(entity)}
    entity2id.update({"O": 0})
    return predict2id, entity2id


class MutiHeadDataset(Dataset):
    def __init__(self, data, tokenizer, predict2id, entity2id, max_len=256, is_train=True):
        super(MutiHeadDataset, self).__init__()
        self.data = data
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

    def get_data_loader(self, batch_size=16, num_workers=0, shuffle=True):
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=self.create_collate_fn())
