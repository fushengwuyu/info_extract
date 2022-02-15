# author: sunshine
# datetime:2021/7/23 上午10:17
import torch
from transformers import BertTokenizerFast
from src.re_gplinker import GPLinker
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from functools import partial
import numpy as np
import json
import random
from argparse import Namespace


class GplinkerDataset(Dataset):
    def __init__(self, data_path, tokenizer, predict2id, max_len=256, is_train=True):
        super().__init__()
        self.data = self.load_data(data_path)[:100]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.predict2id = predict2id
        self.is_train = is_train

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def find_index(self, offset_mapping, index):
        for i, offset in enumerate(offset_mapping):
            if offset[0] <= index < offset[1]:
                return i
        return -1

    def create_collate_fn_sparse(self):

        def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
            """Numpy函数，将序列padding到同一长度
            """
            if length is None:
                length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
            elif not hasattr(length, '__getitem__'):
                length = [length]

            slices = [np.s_[:length[i]] for i in range(seq_dims)]
            slices = tuple(slices) if len(slices) > 1 else slices[0]
            pad_width = [(0, 0) for _ in np.shape(inputs[0])]

            outputs = []
            for x in inputs:
                x = x[slices]
                for i in range(seq_dims):
                    if mode == 'post':
                        pad_width[i] = (0, length[i] - np.shape(x)[i])
                    elif mode == 'pre':
                        pad_width[i] = (length[i] - np.shape(x)[i], 0)
                    else:
                        raise ValueError('"mode" argument must be "post" or "pre".')
                x = np.pad(x, pad_width, 'constant', constant_values=value)
                outputs.append(x)

            return np.array(outputs)

        def collate(examples):
            batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
            texts = [e[0] for e in examples]
            inputs = self.tokenizer(texts, padding='longest', max_length=self.max_len,
                                    truncation='longest_first', return_offsets_mapping=True)
            offset_mapping = inputs['offset_mapping']
            token_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

            if not self.is_train:
                return [token_ids, token_type_ids, attention_mask, offset_mapping, texts, [e[1] for e in examples]]

            for i in range(len(texts)):
                spoes = set()
                for s, p, o in examples[i][1]:
                    sh = self.find_index(offset_mapping[i], s[0])
                    st = self.find_index(offset_mapping[i], s[1])
                    oh = self.find_index(offset_mapping[i], o[0])
                    ot = self.find_index(offset_mapping[i], o[1])
                    p = self.predict2id[p]
                    if sh != -1 and st != -1 and oh != -1 and ot != -1:
                        spoes.add((sh, st, p, oh, ot))
                entity_labels = [set() for _ in range(2)]
                head_labels = [set() for _ in range(len(self.predict2id))]
                tail_labels = [set() for _ in range(len(self.predict2id))]

                for sh, st, p, oh, ot in spoes:
                    entity_labels[0].add((sh, st))
                    entity_labels[1].add((oh, ot))
                    head_labels[p].add((sh, oh))
                    tail_labels[p].add((st, ot))
                for label in entity_labels + head_labels + tail_labels:
                    if not label:  # 至少要有一个标签
                        label.add((0, 0))  # 如果没有则用0填充
                entity_labels = sequence_padding([list(l) for l in entity_labels])
                head_labels = sequence_padding([list(l) for l in head_labels])
                tail_labels = sequence_padding([list(l) for l in tail_labels])
                batch_entity_labels.append(entity_labels)
                batch_head_labels.append(head_labels)
                batch_tail_labels.append(tail_labels)

            batch_entity_labels = sequence_padding(
                batch_entity_labels, seq_dims=2
            )
            batch_head_labels = sequence_padding(
                batch_head_labels, seq_dims=2
            )
            batch_tail_labels = sequence_padding(
                batch_tail_labels, seq_dims=2
            )

            batch_entity_labels = torch.from_numpy(batch_entity_labels)
            batch_head_labels = torch.from_numpy(batch_head_labels)
            batch_tail_labels = torch.from_numpy(batch_tail_labels)

            return [token_ids, token_type_ids, attention_mask, batch_entity_labels, batch_head_labels,
                    batch_tail_labels, texts]

        return partial(collate)

    def load_data(self, path):
        def search(index, patten):
            for i in range(len(index)):
                if index[i:i + len(patten)] == patten:
                    return (i, i + len(patten) - 1)
            return -1

        D = []
        with open(path, 'r', encoding='utf-8') as rd:
            for line in rd:
                d = json.loads(line)
                text = d['text']
                spoes = []
                for spo in d['spo_list']:
                    subject_idx = search(text, spo['subject'])
                    object_idx = search(text, spo['object'])
                    if subject_idx != -1 and object_idx != -1:
                        spoes.append((subject_idx, spo['predicate'], object_idx))
                D.append([text, spoes])

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

        self.predicate2id, self.id2predicate = self.load_mapping(cfg.corpus_path + '/all_50_schemas')

        self.model = GPLinker.from_pretrained(cfg.bert_path, p_classes=len(self.predicate2id))

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

        return predicate2id, id2predicate

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

        best_score = 1e-10
        for epoch in range(int(self.cfg.epoch_num)):

            step = 0
            with tqdm(self.train_loader) as progress_bar:
                for batch in progress_bar:
                    step += 1
                    progress_bar.set_description(f"Epoch {epoch}/{int(self.cfg.epoch_num)}")
                    try:
                        loss = self.training_step(batch)
                        loss = loss / self.cfg.accum_iter

                        loss.backward()
                    except:
                        print(batch[-1])
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)

                    if (step % self.cfg.accum_iter == 0) or (step == len(self.train_loader)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.schedule.step()

                    progress_bar.set_postfix(loss=loss.item())

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

        batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch[:-1])
        input_ids, token_type_ids, attention_mask, el, hl, tl = batch
        loss = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=[el, hl, tl])
        return loss

    def validation_step(self, batch):
        A, B, C = 1e-12, 1e-12, 1e-12
        offset_mapping, texts, spoes_gold = batch[3:]
        batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch[:3])
        input_ids, token_type_ids, attention_mask = batch

        y_pred = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        entity_pred, head_pred, tail_pred = y_pred
        for entity, head, tail, m, gold in zip(entity_pred.detach().cpu().numpy(), head_pred.detach().cpu().numpy(),
                                               tail_pred.detach().cpu().numpy(), offset_mapping, spoes_gold):
            subjects, objects = set(), set()
            entity[:, [0, -1]] -= np.inf
            entity[:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(entity > 0)):
                if l == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))

            spoes = set()
            for sh, st in subjects:
                for oh, ot in objects:
                    p1s = np.where(head[:, sh, oh] > 0)[0]
                    p2s = np.where(tail[:, st, ot] > 0)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        spoes.add((m[sh][0], m[st][-1], p, m[oh][0], m[ot][-1]))

            gold = [(g[0][0], g[0][1] + 1, self.predicate2id[g[1]], g[2][0], g[2][1] + 1) for g in gold]
            A += len(set(spoes) & set(gold))
            B += len(set(spoes))
            C += len(set(gold))

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

        train_dataset = GplinkerDataset(self.cfg.corpus_path + '/sample.json', self.tokenizer, self.predicate2id,
                                        is_train=True)
        dev_dataset = GplinkerDataset(self.cfg.corpus_path + '/dev_data.json', self.tokenizer, self.predicate2id,
                                      is_train=False)
        # 若并行加载数据，不可用于调试
        if self.debug:
            num_workers = 0
        else:
            num_workers = 12
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=train_dataset.create_collate_fn_sparse(), num_workers=num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                                collate_fn=dev_dataset.create_collate_fn_sparse(), num_workers=num_workers)
        return train_loader, dev_loader


if __name__ == '__main__':
    cfg = {
        "bert_path": "/home/sunshine/pre_models/pytorch/bert-base-chinese",
        "corpus_path": "/home/sunshine/python/Info_extract/dataset/re",
        "learning_rate": 2e-5,
        "batch_size": 1,
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
    trainer.fit()
    # trainer.resume('/home/sunshine/python/Info_extract/models/pytorch_GPLinker.bin')
    # result = trainer.evaluate()
    # print(result)
    # for batch in trainer.train_loader:
    #     ...
