import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AlbertForSequenceClassification,
    MT5ForConditionalGeneration,
    AutoModelWithLMHead,
    AutoTokenizer,
    AlbertTokenizer,
    T5Tokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import get_linear_schedule_with_warmup


class AntibioticsDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
            )

        return {
            'text':text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label,
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = AntibioticsDataset(
        text=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
        )

def eval_model(model, data_loader, device, best_acc, writer, step_t, metrics=False):
    model.eval()

    preds = []
    losses = []
    label = []

    with torch.no_grad():
        eval_progress = tqdm(enumerate(data_loader))
        for step,d in eval_progress:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels=labels
                )

            _, predicted = torch.max(outputs.logits, 1)
            preds.append(predicted)

            loss = outputs.loss
            losses.append(loss.item())
            label.append(labels)

        preds = torch.cat(preds, dim=0)
        label = torch.cat(label, dim=0)

        accuracy = accuracy_score(label.detach().cpu().clone().numpy(), preds.detach().cpu().clone().numpy())

        if metrics:
            precision, recall, f1, _ = precision_recall_fscore_support(label.detach().cpu().clone().numpy(), preds.detach().cpu().clone().numpy(), average='binary')
            print("Validation precision: ",precision)
            print("Validation recall: ",recall)
            print("Validation f1: ",f1)
            print("Validation accuracy: ",accuracy)

        if accuracy>best_acc:
            best_acc = accuracy
            if not args.multi:
                torch.save(model.state_dict(), "../models/"+ args.model + "_ft.pt")

        writer.add_scalar('loss/test', np.mean(losses), step_t)
        writer.add_scalar('accuracy', accuracy, step_t)

    return best_acc

def train_epoch(
    model,
    data_loader,
    optimizer,
    device,
    scheduler,
    writer,
    e,
    best_acc
):
    model = model.train()
    l = len(data_loader)
    losses = []

    plot_step = 0
    train_progress = tqdm(enumerate(data_loader, 0))

    for step,d in train_progress:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels=labels
        )
        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


        if step % int(l/2)==0 and step != 0:
            best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e*2 + plot_step)
            writer.add_scalar('loss/training', np.mean(losses), e*2 + plot_step)
            plot_step+=1


        train_progress.set_description(
            '| Train_loss: {:.3f}'.format(loss)
        )
    return best_acc


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Huggingface modetl', required=True)
parser.add_argument('--epochs', help='Trainng epochs', default = 15, type=int)
parser.add_argument('--warmup_steps', help='Warmup steps', default = 500, type=int)
parser.add_argument('--learning_rate', help='Learning rate', default = 5e-5, type=float)
parser.add_argument('--weight_decay', help='Weight decay', default = 0.01, type=float)
parser.add_argument('--eval', help='Evaluation mode', action="store_true")
parser.add_argument('--continue_training', help='Keep training', action="store_true")
parser.add_argument('--multi', help='Multiple runs', action="store_true")

args = parser.parse_args()


if not args.multi:
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    print("Seed")
else:
    print('Multiple runs')

df = pd.read_csv("../data/dataset_no_recipe.csv")
df.columns = ['text', 'label']
random = df.iloc[np.random.permutation(len(df))]
train = random.iloc[:round(len(df)*.8)]
test = random.iloc[round(len(df)*.8):]
print(train.shape)
print(test.shape)

if args.model == 'KB/albert-base-swedish-cased-alpha':
    tokenizer = AlbertTokenizer.from_pretrained('KB/albert-base-swedish-cased-alpha')
    model = AlbertForSequenceClassification.from_pretrained('KB/albert-base-swedish-cased-alpha')
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('../logs/' + args.model)


max_len = 512
train_batch_size = 16
test_batch_size = 64

train_data_loader = create_data_loader(train, tokenizer, max_len, train_batch_size)
test_data_loader = create_data_loader(test, tokenizer, max_len, test_batch_size)


if args.eval:
    model.load_state_dict(torch.load("../models/"+ args.model + "_ft.pt"))
    model = model.to(device)

    best_acc = 0
    e = 0
    best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e, metrics=True)
    num_parameters = sum(p.numel() for p in model.parameters())
    print('Number of parameters: ', num_parameters)
    exit()

model = model.to(device)

if args.continue_training:
    model.load_state_dict(torch.load("../models/"+ args.model + ".pt"))
    args.warmup_steps = 0

epochs = args.epochs
total_steps = len(train_data_loader)*epochs

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps)

epoch_progress = tqdm(range(epochs))

acc = 0
for e in epoch_progress:
    epoch_progress.set_description(
        'Epoch:{}  '.format(e)
    )
    acc = train_epoch(model, train_data_loader, optimizer, device, scheduler, writer, e, acc)

best_acc = 0
best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e, metrics=True)
