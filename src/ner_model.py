# author: sunshine
# datetime:2021/7/23 上午10:17
import torch
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


class Net(BertPreTrainedModel):
    def __init__(self, config, head_type, num_class, max_len=256):
        super(Net, self).__init__(config)
        self.bert = BertModel(config)

        if head_type == 'GlobalPointer':

            self.head = GlobalPointer(heads=num_class, head_size=64, hidden_size=config.hidden_size)

        elif head_type == 'Biaffine':
            self.head = Biaffine(in_size=config.hidden_size, out_size=num_class, Position=False)

        elif head_type == 'Mutihead':
            self.head = MutiHeadSelection(hidden_size=config.hidden_size, c_size=num_class, re_position=True,
                                          max_len=max_len)

        elif head_type == 'TxMutihead':
            self.head = TxMutihead(hidden_size=config.hidden_size, c_size=num_class, abPosition=True, maxlen=max_len)
        elif head_type == 'EfficientGlobalPointer':
            self.head = EfficientGlobalPointer(heads=num_class, head_size=64, hidden_size=config.hidden_size)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        x = x.last_hidden_state
        logits = self.head(x, mask=attention_mask)
        if label is not None:
            loss = self.global_pointer_crossentropy(label, logits)
            return loss
        else:
            return logits

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """
        多标签交叉熵
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def global_pointer_crossentropy(self, y_true, y_pred):
        """结合GlobalPointer设计的交叉熵
        """
        bh = y_pred.shape[0] * y_pred.shape[1]
        y_true = torch.reshape(y_true, (bh, -1))
        y_pred = torch.reshape(y_pred, (bh, -1))
        return torch.mean(self.multilabel_categorical_crossentropy(y_true, y_pred))


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)

    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)

    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


def relative_position_encoding(depth, max_length=512, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(max_length)
    range_mat = range_vec.repeat(max_length).view(max_length, max_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    embeddings_table = torch.zeros(vocab_size, depth)
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    positions_encoding = positions_encoding.view(my_shape)
    return positions_encoding


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class GlobalPointer(nn.Module):
    def __init__(self, heads, head_size, hidden_size, RoPE=True, tril_mask=True):
        super(GlobalPointer, self).__init__()
        self.tril_mask = tril_mask
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, self.head_size * self.heads * 2)

    def forward(self, inputs, mask=None, ):
        inputs = self.dense(inputs)

        inputs = torch.split(inputs, self.head_size * 2, dim=-1)
        inputs = torch.stack(inputs, dim=-2)

        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)

            """
            cos_pos = K.repeat_elements(pos[..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(pos[..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
            """
            cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
            sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum('bmhd, bnhd->bhmn', qw, kw)

        # 排除padding  排除下三角
        if self.tril_mask:
            logits = add_mask_tril(logits, mask)
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(EfficientGlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size, head_size * 2, bias=True)
        self.linear_2 = nn.Linear(head_size * 2, heads * 2, bias=True)

    def forward(self, inputs, mask=None):
        inputs = self.linear_1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., 1::2].repeat(1, 1, 2)
            sin_pos = pos[..., ::2].repeat(1, 1, 2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.head_size ** 0.5
        bias = torch.einsum('bnh -> bhn', self.linear_2(inputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # 排除padding跟下三角
        logits = add_mask_tril(logits, mask)
        return logits

class MutiHeadSelection(nn.Module):
    def __init__(self, hidden_size, c_size, ab_position=False, re_position=False, max_len=512, max_relative=127):
        super(MutiHeadSelection, self).__init__()
        self.hidden_size = hidden_size
        self.c_size = c_size
        self.ab_position = ab_position
        self.re_position = re_position
        self.Wh = nn.Linear(hidden_size * 2, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size, self.c_size)

        if self.re_position:
            self.relative_positions_encoding = relative_position_encoding(depth=2 * hidden_size,
                                                                          max_length=max_len,
                                                                          max_relative_position=max_relative)

    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        if self.ab_position:
            # 由于是加性拼接,无法使用RoPE，因此直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(self.hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1], dim=-1)
        if self.re_position:
            relations_keys = self.relative_positions_encoding[:input_length, :input_length, :].to(inputs.device)
            concat_x += relations_keys
        hij = torch.tanh(self.Wh(concat_x))
        logits = self.Wo(hij)
        logits = logits.permute(0, 3, 1, 2)
        logits = add_mask_tril(logits, mask)
        return logits


class TxMutihead(nn.Module):

    def __init__(self, hidden_size, c_size, abPosition=False, rePosition=False, maxlen=None, max_relative=127):
        super(TxMutihead, self).__init__()
        self.hidden_size = hidden_size
        self.c_size = c_size
        self.abPosition = abPosition
        self.rePosition = rePosition
        self.Wh = nn.Linear(hidden_size * 4, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size, self.c_size)
        if self.rePosition:
            self.relative_positions_encoding = relative_position_encoding(max_length=maxlen,
                                                                          depth=4 * hidden_size,
                                                                          max_relative_position=max_relative)

    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        if self.abPosition:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(self.hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1, x2 - x1, x2.mul(x1)], dim=-1)
        if self.rePosition:
            relations_keys = self.relative_positions_encoding[:input_length, :input_length, :].to(inputs.device)
            concat_x += relations_keys
        hij = torch.tanh(self.Wh(concat_x))
        logits = self.Wo(hij)
        logits = logits.permute(0, 3, 1, 2)
        logits = add_mask_tril(logits, mask)
        return logits


class Biaffine(nn.Module):

    def __init__(self, in_size, out_size, Position=False):
        super(Biaffine, self).__init__()
        self.out_size = out_size
        self.weight1 = Parameter(torch.Tensor(in_size, out_size, in_size))
        self.weight2 = Parameter(torch.Tensor(2 * in_size + 1, out_size))
        self.Position = Position
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        hidden_size = inputs.shape[-1]
        if self.Position:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1], dim=-1)
        concat_x = torch.cat([concat_x, torch.ones_like(concat_x[..., :1])], dim=-1)
        # bxi,oij,byj->boxy
        logits_1 = torch.einsum('bxi,ioj,byj -> bxyo', inputs, self.weight1, inputs)
        logits_2 = torch.einsum('bijy,yo -> bijo', concat_x, self.weight2)
        logits = logits_1 + logits_2
        logits = logits.permute(0, 3, 1, 2)
        logits = add_mask_tril(logits, mask)
        return logits
