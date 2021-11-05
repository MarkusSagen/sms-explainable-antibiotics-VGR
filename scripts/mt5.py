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
    MT5ForConditionalGeneration,
    AutoModelWithLMHead,
    AutoTokenizer,
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

        encoding_labels = self.tokenizer.encode_plus(
            label,
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
            'labels_ids': encoding_labels['input_ids'].flatten()
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
    predictions = []
    actuals = []
    losses = []

    with torch.no_grad():
        eval_progress = tqdm(enumerate(data_loader))

        for step,d in eval_progress:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels_ids"].to(device)

            generated_ids = model.generate(input_ids = input_ids, attention_mask=attention_mask)

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in labels]

            predictions.extend(preds)
            actuals.extend(target)

            outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels=labels
            )

            loss = outputs.loss
            losses.append(loss.item())


        if metrics:
            predictions_int = [1 if p=='Positive' else 0 for p in predictions]
            actuals_int = [1 if a=='Positive' else 0 for a in actuals]
            precision, recall, f1, _ = precision_recall_fscore_support(actuals_int, predictions_int, average='binary')
        accuracy = accuracy_score(actuals, predictions)

        eval_progress.set_description(
            'Eval_loss: {:.6f} | Eval_accuracy: {:.3f}'.format(np.mean(losses), accuracy)
        )

        if accuracy>best_acc:
            best_acc = accuracy
            if not args.multi:
                torch.save(model.state_dict(), "../models/mt5.pt")

        #print("Validation accuracy: ",accuracy)
        if metrics:
            print("Validation precision: ",precision)
            print("Validation recall: ",recall)
            print("Validation f1: ",f1)
            print("Validation accuracy: ",accuracy)


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

    accumulation_steps = args.accumulation
    train_progress = tqdm(enumerate(data_loader, 0))
    plot_step = 0

    losses = []

    for step,d in train_progress:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels_ids"].to(device)
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels=labels
        )
        loss = outputs.loss

        loss = loss / accumulation_steps
        losses.append(loss.item())
        loss.backward()

        if (step+1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        if step % (int(l/2)-1) ==0 and step != 0:
            #best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e * l +step)
            best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e*2 + plot_step)
            #writer.add_scalar('training loss', loss, e * l +step)
            writer.add_scalar('loss/training', np.mean(losses)*accumulation_steps, e*2 + plot_step)
            plot_step += 1

        train_progress.set_description(
            'Train_loss: {:.6f}'.format(np.mean(losses))
        )
    return best_acc

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Trainng epochs', default = 40, type=int)
parser.add_argument('--warmup_steps', help='Warmup steps', default = 0, type=int)
parser.add_argument('--learning_rate', help='Learning rate', default = 10e-5, type=float)
parser.add_argument('--weight_decay', help='Weight decay', default = 0, type=float)
parser.add_argument('--accumulation', help='Gradient accumulation', default = 4, type=int)
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

df = pd.read_csv("../data/seq_dataset.csv")
df.columns = ['text', 'label']
random = df.iloc[np.random.permutation(len(df))]
train = random.iloc[:round(len(df)*.8)]
test = random.iloc[round(len(df)*.8):]
print(train.shape)
print(test.shape)

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('../logs/mt5_lr')

max_len = 512
train_batch_size = 4
test_batch_size = 4

train_data_loader = create_data_loader(train, tokenizer, max_len, train_batch_size)
test_data_loader = create_data_loader(test, tokenizer, max_len, test_batch_size)

if args.eval:
    model.load_state_dict(torch.load("../models/mt5.pt"))
    model = model.to(device)

    best_acc = 0
    e = 0
    best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e, metrics=True)
    num_parameters = sum(p.numel() for p in model.parameters())
    print('Number of parameters: ', num_parameters)
    exit()

model = model.to(device)

if args.continue_training:
    model.load_state_dict(torch.load("../models/mt5.pt"))

epochs = args.epochs
total_steps = len(train_data_loader)*epochs

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)#0.001)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps)

epoch_progress = tqdm(range(epochs))
acc=0
for e in epoch_progress:
    epoch_progress.set_description(
        'Epoch:{}  '.format(e)
    )
    acc = train_epoch(model, train_data_loader, optimizer, device, scheduler, writer, e, acc)

best_acc = 0
best_acc = eval_model(model,test_data_loader, device, best_acc, writer, e, metrics=True)
