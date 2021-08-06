# author: sunshine
# datetime:2021/7/23 上午10:17
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from src.re.muti_head.spo_net import SPOBiaffine
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, tokenizer, id2p, id2e):

        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

        self.model = SPOBiaffine(args, len(id2p), len(id2e))
        self.id2p = id2p
        self.model.to(self.device)
        if args.train_mode == "eval":
            self.resume()

        self.train_dataloader, self.dev_dataloader = data_loaders

        # 设置优化器，优化策略
        train_steps = (len(self.train_dataloader) / args.batch_size) * args.epoch_num
        self.optimizer, self.schedule = self.set_optimizer(args=args,
                                                           model=self.model,
                                                           train_steps=train_steps)

        self.loss_bce = torch.nn.BCELoss()

    def set_optimizer(self, args, model, train_steps=None):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # optimizer, num_warmup_steps, num_training_steps
        schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=train_steps
        )
        return optimizer, schedule

    def train(self):

        best_f1 = 0.0
        self.model.train()
        step_gap = 5
        step_eval = 10
        for epoch in range(int(self.args.epoch_num)):
            for step, batch in tqdm(enumerate(self.train_dataloader)):

                loss = self.forward(batch, is_eval=False)
                if step % step_gap == 0:
                    info = "step {} / {} of epoch {}, train/loss: {}"
                    print(info.format(step, self.train_dataloader, epoch, loss.item()))

                if (step + 1) % step_eval == 0:

                    p, r, f1 = self.evaluate(self.dev_dataloader)
                    print("p: {}, r: {}, f1: {}".format(p, r, f1))
                    if f1 >= best_f1:
                        best_f1 = f1

                        # 保存模型
                        self.save()

    def forward(self, batch, is_eval=False):
        batch = tuple(t.to(self.device) for t in batch)
        if not is_eval:
            input_ids, attention_mask, token_type_ids, start_label, end_label, entity_type, r_label = batch
            self.optimizer.zero_grad()
            start_logits, end_logits, r_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                type_ids=token_type_ids,
                cs_ids=entity_type)

            start_loss = self.loss_bce(start_logits, start_label.float())
            end_loss = self.loss_bce(end_logits, end_label.float())
            r_loss = self.loss_bce(r_logits, r_label.float())

            loss = 10 * start_loss + 10 * end_loss + r_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.warmup_proportion)
            self.optimizer.step()
            self.schedule.step()

            return loss
        else:
            """
            返回 [(i, j), p, (i, j)]
            """
            answers = []
            input_ids, attention_mask, token_type_ids = batch
            batch_subject, batch_end_list, out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                type_ids=token_type_ids)

            out_logits = out.cpu().numpy()
            for entities, end_list, out in zip(batch_subject, batch_end_list, out_logits):
                answer = []
                s_end, o_end, predict = np.where(out > 0.5)
                for _s, _o, p in zip(s_end, o_end, predict):
                    if _s in end_list and _o in end_list:
                        sub = entities[end_list.index(_s)]
                        obj = entities[end_list.index(_o)]
                        answer.append((sub, p, obj))

                answers.append(answer)
            return answers

    def resume(self):
        resume_model_file = self.args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def save(self):
        logger.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        output_model_file = self.args.output + "/pytorch_model.bin"
        torch.save(model_to_save.state_dict(), str(output_model_file))

    def evaluate(self, dataloader):
        """验证
        """
        self.model.eval()
        A, B, C = 1e-10, 1e-10, 1e-10
        with torch.no_grad():
            for batch in tqdm(dataloader):
                gold_answers, texts, offset_mapping = batch[3:]
                answers = self.forward(batch=batch[:3], is_eval=True)
                for gold, answer, text, mapping in zip(gold_answers, answers, texts, offset_mapping):
                    format_answer = []
                    for ans in answer:
                        s, p, o = ans
                        sub = text[mapping[s[0]][0]: mapping[s[1]][1]]
                        obj = text[mapping[o[0]][0]: mapping[o[1]][1]]
                        format_answer.append((sub, self.id2p[p], obj))

                    A += len(set(gold) & set(format_answer))
                    B += len(set(gold))
                    C += len(set(format_answer))

        self.model.train()
        return A / C, A / B, 2 * A / (B + C)  # p, r, f1


if __name__ == '__main__':
    input = torch.randn((4, 10, 3))
    target = torch.zeros((4, 10, 3), dtype=torch.float)
    target[0, 3, 1] = 1
    target[0, 4, 2] = 1
    target[0, 5, 0] = 1
    target[0, 6, 2] = 1
    s_input = input.sigmoid()
    loss = F.binary_cross_entropy(s_input, target)
    print(loss)
