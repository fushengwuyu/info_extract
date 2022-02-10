# author: sunshine
# datetime:2022/2/10 下午4:42
import torch
from transformers import BertModel, BertPreTrainedModel
from src.ner_model import GlobalPointer
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


class REModel(BertPreTrainedModel):
    def __init__(self, config, p_classes):
        super(REModel, self).__init__(config)
        self.bert = BertModel(config)
        self.entity = GlobalPointer(2, head_size=64, hidden_size=config.hidden_size)
        self.head = GlobalPointer(p_classes, head_size=64, hidden_size=config.hidden_size, RoPE=False, tril_mask=False)
        self.tail = GlobalPointer(p_classes, head_size=64, hidden_size=config.hidden_size, RoPE=False, tril_mask=False)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        x = x.last_hidden_state

        o1 = self.entity(x)
        o2 = self.head(x)
        o3 = self.tail(x)

        if label is not None:
            l1, l2, l3 = label
            loss1 = self.globalpointer_crossentropy(l1, o1)
            loss2 = self.globalpointer_crossentropy(l2, o2)
            loss3 = self.globalpointer_crossentropy(l3, o3)
            return loss1 + loss2 + loss3
        else:
            return o1, o2, o3

    def globalpointer_crossentropy(self, y_true, y_pred):
        """给GlobalPointer设计的交叉熵
            """
        shape = K.shape(y_pred)
        y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
        y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
        loss = self.sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
        return K.mean(K.sum(loss, axis=1))

    def sparse_multilabel_categorical_crossentropy(self, y_true, y_pred, mask_zero=False):
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
        zeros = K.zeros_like(y_pred[..., :1])
        y_pred = K.concatenate([y_pred, zeros], axis=-1)
        if mask_zero:
            infs = zeros + K.infinity()
            y_pred = K.concatenate([infs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = batch_gather(y_pred, y_true)
        y_pos_1 = K.concatenate([y_pos_2, zeros], axis=-1)
        if mask_zero:
            y_pred = K.concatenate([-infs, y_pred[..., 1:]], axis=-1)
            y_pos_2 = batch_gather(y_pred, y_true)
        pos_loss = K.logsumexp(-y_pos_1, axis=-1)
        all_loss = K.logsumexp(y_pred, axis=-1)
        aux_loss = K.logsumexp(y_pos_2, axis=-1) - all_loss
        aux_loss = K.clip(1 - K.exp(aux_loss), K.epsilon(), 1)
        neg_loss = all_loss + K.log(aux_loss)
        return pos_loss + neg_loss
