# author: sunshine
# datetime:2022/2/10 下午4:42
import torch
from transformers import BertModel, BertPreTrainedModel
from src.ner_model import GlobalPointer
import numpy as np
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


class GPLinker(BertPreTrainedModel):
    def __init__(self, config, p_classes):
        super(GPLinker, self).__init__(config)
        self.bert = BertModel(config)
        self.entity = GlobalPointer(2, head_size=64, hidden_size=config.hidden_size)
        self.head = GlobalPointer(p_classes, head_size=64, hidden_size=config.hidden_size, RoPE=False, tril_mask=False)
        self.tail = GlobalPointer(p_classes, head_size=64, hidden_size=config.hidden_size, RoPE=False, tril_mask=False)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        x = x.last_hidden_state

        o1 = self.entity(x, mask=attention_mask)
        o2 = self.head(x)
        o3 = self.tail(x)

        if labels is not None:
            l1, l2, l3 = labels
            loss1 = self.mutilabel_crossentropy(l1, o1, is_sparse=True)
            loss2 = self.mutilabel_crossentropy(l2, o2, is_sparse=True)
            loss3 = self.mutilabel_crossentropy(l3, o3, is_sparse=True)
            return loss1 + loss2 + loss3
        else:
            return o1, o2, o3

    def mutilabel_crossentropy(self, y_true, y_pred, is_sparse=False):
        if is_sparse:
            loss = self.sparse_global_pointer_crossentropy(y_true, y_pred)
        else:
            loss = self.global_pointer_crossentropy(y_true, y_pred)
        return loss

    def global_pointer_crossentropy(self, y_true, y_pred):
        """结合GlobalPointer设计的交叉熵
        """
        bh = y_pred.shape[0] * y_pred.shape[1]
        y_true = torch.reshape(y_true, (bh, -1))
        y_pred = torch.reshape(y_pred, (bh, -1))

        # 下面是多标签交叉熵
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        return torch.mean(loss)

    def sparse_global_pointer_crossentropy(self, y_true, y_pred, mask_zero=True):
        """稀疏版多标签分类的交叉熵
        说明：
            1. y_true.shape=[..., num_positive]，
               y_pred.shape=[..., num_classes]；
            2. 请保证y_pred的值域是全体实数，换言之一般情况下
               y_pred不用加激活函数，尤其是不能加sigmoid或者
               softmax；
            3. 预测阶段则输出y_pred大于0的类；
            4. 详情请看：https://kexue.fm/archives/7359 。
        """

        """给GlobalPointer设计的交叉熵
                    """
        shape = y_pred.shape
        y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
        y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))

        # 下面才是稀疏交叉熵
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if mask_zero:
            infs = zeros + 1e12
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), 1e-7, 1)
        neg_loss = all_loss + torch.log(aux_loss)

        loss = neg_loss + pos_loss

        return torch.mean(torch.sum(loss, dim=1))
