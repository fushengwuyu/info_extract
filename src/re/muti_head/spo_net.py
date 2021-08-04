# author: sunshine
# datetime:2021/7/28 上午11:19
import torch
from transformers import BertModel
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, max_len):
        super(Biaffine, self).__init__()
        self.max_len = max_len
        self.w1 = Parameter(torch.Tensor(in_size, out_size, in_size))
        self.w2 = Parameter(torch.Tensor(2 * in_size + 1, out_size))

    def forward(self, input1, input2):
        """
        input1: (batch, seq_len, dim)
        input2: (batch, seq_len, dim)
        """
        seq_len = input1.shape[1]

        f1 = torch.unsqueeze(input1, 2)
        f2 = torch.unsqueeze(input2, 1)

        f1 = f1.repeat(1, 1, seq_len, 1)  # [batch, max_len, max_len, dim]
        f2 = f2.repeat(1, seq_len, 1, 1)

        concat = torch.cat([f1, f2, torch.ones_like(f1[..., :1])], dim=-1)  # [batch, max_len, max_len, 2*dim + 1]

        logits1 = torch.einsum('bxi, ioj, byj->bxyo', input1, self.w1, input2)  # [batch, max_len, max_len, p_num]
        logits2 = torch.einsum('bijy, yo->bijo', concat, self.w2)  # [batch, max_len, max_len, p_num]

        return logits1 + logits2


class SPOMutiHead(nn.Module):
    def __init__(self, args, p_num, e_num):
        super(SPOMutiHead, self).__init__()
        self.max_len = args.max_len
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.start_fc = nn.Sequential(
            nn.Linear(128, e_num),
            nn.Sigmoid()
        )
        self.end_fc = nn.Sequential(
            nn.Linear(128, e_num),
            nn.Sigmoid()
        )

        self.cs_emb = nn.Embedding(e_num, 64)
        self.fc1 = nn.Linear(768 + 64, 128)
        self.fc2 = nn.Linear(768 + 64, 128)

        self.seq = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, p_num),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, type_ids, cs_ids):
        x0 = self.bert(input_ids, attention_mask, type_ids)
        share = x0.last_hidden_state

        start_logits = self.start_fc(share)
        start_logits = start_logits ** 2

        end_logits = self.end_fc(share)
        end_logits = end_logits ** 2

        entity_emb = self.cs_emb(cs_ids)
        concat = torch.cat([share, entity_emb], dim=-1)

        f1 = self.fc1(concat)
        f2 = self.fc2(concat)

        f1 = torch.unsqueeze(f1, 1)
        f2 = torch.unsqueeze(f2, 2)

        f1 = f1.repeat(1, self.max_len, 1, 1)
        f2 = f2.repeat(1, 1, self.max_len, 1)

        concat_f = torch.cat([f1, f2], dim=-1)

        out = self.seq(concat_f)
        out = out ** 4

        return start_logits, end_logits, out


class SPOBiaffine(nn.Module):
    def __init__(self, args, p_num, e_num):
        super(SPOBiaffine, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.start_dense = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, e_num),
            nn.Sigmoid()
        )
        self.end_dense = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, e_num),
            nn.Sigmoid()
        )
        self.emb = nn.Embedding(e_num, 100)
        self.f1_dense = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.f2_dense = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.biaffine = Biaffine(128 + 100, p_num, args.max_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, type_ids, cs_ids=None):
        x = self.bert(input_ids, attention_mask, type_ids, output_hidden_states=True)
        hidden_states = x.hidden_states
        layer_1 = hidden_states[-1]
        layer_2 = hidden_states[-2]

        start_logits = self.start_dense(layer_1)
        start_logits = start_logits ** 2

        end_logits = self.end_dense(layer_1)
        end_logits = end_logits ** 2

        if cs_ids is not None:
            cs_emb = self.emb(cs_ids)
            concat_cs = torch.cat([layer_1, layer_2], dim=-1)

            f1 = self.f1_dense(concat_cs)
            f1 = torch.cat([f1, cs_emb], dim=-1)

            f2 = self.f2_dense(concat_cs)
            f2 = torch.cat([f2, cs_emb], dim=-1)

            biaffine_layer = self.biaffine(f1, f2)
            out = self.sigmoid(biaffine_layer)
            out = out ** 4
            return start_logits, end_logits, out
        else:
            # 推理
            batch_subject, batch_end_list, cs_ids = self.extract_type(start_logits, end_logits)
            cs_emb = self.emb(cs_ids)
            concat_cs = torch.cat([layer_1, layer_2], dim=-1)

            f1 = self.f1_dense(concat_cs)
            f1 = torch.cat([f1, cs_emb], dim=-1)

            f2 = self.f2_dense(concat_cs)
            f2 = torch.cat([f2, cs_emb], dim=-1)

            biaffine_layer = self.biaffine(f1, f2)
            out = self.sigmoid(biaffine_layer)
            out = out ** 4
            return batch_subject, batch_end_list, out

    def extract_type(self, start_logits, end_logits):

        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()
        cs_ids = []
        batch_subject, batch_end_list = [], []
        seq_len = start_logits.shape[1]
        for start, end in zip(start_logits, end_logits):
            s, st = np.where(start > 0.5)
            e, et = np.where(end > 0.5)

            entities, end_list = [], []
            s_stop = np.zeros((seq_len, ))

            for i, t in zip(s, st):
                j = e[e >= i]
                et = et[e >= i]
                if len(j) > 0 and et[0] == t:
                    _j = j[0]
                    end_list.append(_j)
                    entities.append((i, _j))
                    s_stop[0][_j] = t

            cs_ids.append(s_stop)
            batch_subject.append(entities)
            batch_end_list.append(end_list)
        cs_ids = torch.tensor(cs_ids, dtype=torch.long)
        return batch_subject, batch_end_list, cs_ids

if __name__ == '__main__':

    biaffine = Biaffine(10, 4, 128)

    f1 = torch.rand((4, 16, 10))
    f2 = torch.rand((4, 16, 10))

    o = biaffine(f1, f2)
    print(o.shape)