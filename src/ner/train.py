# author: sunshine
# datetime:2021/7/23 上午10:17
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from src.ner.net import Net
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, tokenizer, label2id):

        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

        self.model = Net(args, len(label2id))

        self.model.to(self.device)
        if args.train_mode == "eval":
            self.resume()

        self.train_dataloader, self.dev_dataloader = data_loaders

        # 设置优化器，优化策略
        train_steps = (len(self.train_dataloader) / args.batch_size) * args.epoch_num
        self.optimizer, self.schedule = self.set_optimizer(args=args,
                                                           model=self.model,
                                                           train_steps=train_steps)

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

    def train(self, args):

        best_f1 = 0.0
        self.model.train()
        step_gap = 10
        step_eval = 500
        for epoch in range(int(args.epoch_num)):
            for step, batch in tqdm(enumerate(self.train_dataloader)):

                loss = self.forward(batch, is_eval=False)
                if step % step_gap == 0:
                    print(u"step {} / {} of epoch {}, train/loss: {}".format(step,
                                                                             len(self.train_dataloader) / args.batch_size,
                                                                             epoch, loss.item()))

                if (step + 1) % step_eval == 0:

                    acc = self.evaluate(self.dev_dataloader)
                    print("acc: {}".format(acc))
                    if acc >= best_f1:
                        best_f1 = acc

                        # 保存模型
                        self.save()

    def forward(self, batch, is_eval=False):
        batch = tuple(t.to(self.device) for t in batch)
        if not is_eval:
            input_ids, attention_mask, token_type_ids, label = batch
            self.optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label)

            loss = logits[1]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.warmup_proportion)
            # loss = loss.item()
            self.optimizer.step()
            self.schedule.step()

            return loss
        else:
            input_ids, attention_mask, token_type_ids, label = batch
            y_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            y_pred = torch.greater(y_pred, 0)
            temp_n = torch.sum(label * y_pred).item()
            temp_d = torch.sum(label + y_pred).item()
            return temp_n, temp_d

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
        numerate, denominator = 0, 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                temp_n, temp_d = self.forward(batch=batch, is_eval=True)
                numerate += temp_n
                denominator += temp_d
        self.model.train()
        return 2 * numerate / denominator
