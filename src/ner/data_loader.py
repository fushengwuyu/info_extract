# author: sunshine
# datetime:2021/7/23 上午10:18
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial

"""
样本例子：
（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。|||3    7    pro|||9    13    pro|||16    33    dis|||
"""


def load_data(path):
    """加载数据
    """
    D = []
    labels = set()
    with open(path, 'r', encoding='utf-8') as rd:
        for line in rd:
            fields = line.strip().split('|||')[:-1]
            text = fields[0]
            entities = []
            for info in fields[1:]:
                s, e, l = info.split()
                labels.add(l)
                entities.append((int(s), int(e), l))
            D.append((text, entities))

    return D, labels


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=256):
        super(NERDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

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
            texts = [e[0] for e in examples]
            inputs = self.tokenizer(texts, padding='longest', max_length=self.max_len, truncation='longest_first',
                                    return_offsets_mapping=True)
            token_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

            offset_mapping = inputs['offset_mapping']

            labels = [e[1] for e in examples]
            label_span = torch.zeros((len(texts), len(self.label2id), len(offset_mapping[0]), len(offset_mapping[0])),
                                     dtype=torch.long)
            for i in range(len(texts)):

                for l in labels[i]:
                    """
                    labels: (3, 7, pro)
                    offset_mapping: [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 0)]
                    """
                    s = find_index(offset_mapping[i], l[0])
                    e = find_index(offset_mapping[i], l[1])

                    label_span[i, self.label2id[l[2]], s, e] = 1

            return token_ids, attention_mask, token_type_ids, label_span

        return partial(collate)

    def get_data_loader(self, batch_size=16, num_workers=0, shuffle=True):
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=self.create_collate_fn())


def main():
    train = load_data('../../dataset/ner/train_data.txt')
    dev = load_data('../../dataset/ner/val_data.txt')
    print(len(train), len(dev))
    labels = []
    for e in train + dev:
        for l in e[1]:
            labels.append(l[2])

    from collections import Counter

    count = Counter(labels)
    print(count.most_common())
    # [('bod', 23580), ('dis', 20778), ('sym', 16399), ('pro', 8389), ('dru', 5370), ('ite', 3504), ('mic', 2492), ('equ', 1126), ('dep', 458)]


if __name__ == '__main__':
    # main()

    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('/home/sunshine/pre_models/pytorch/bert-base-chinese')

    a = tokenizer(['我是中国人', "我爱你"], return_offsets_mapping=True)
    print(a['offset_mapping'])
