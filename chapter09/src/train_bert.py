'''
89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
'''
import wandb
import torch
import timm.scheduler
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    )

from make_dataset import make_dataset_for_collate_fn, make_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(model, dataset):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for batch in dataset:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            labels = label_filter(labels)
            outputs = model(inputs, labels=labels)
            loss += outputs.loss.item()
            pred = torch.argmax(outputs.logits, dim=1)
            print(pred)
            print(labels)
            total += len(batch['labels'])
            correct += (pred == labels).sum().item()
        acc = correct / total
        eval_loss = loss / len(dataset)

    return acc, eval_loss

def label_filter(labels):
    new_labels = []
    for label in labels:
        label = label[torch.where(label >= 0)]
        new_labels.append(label)
    
    # newlabels を torch.tensor に変換しつつ、結合する
    new_labels = torch.cat(new_labels, dim=0)

    return new_labels.to(device)

def main():
    wandb.init(project='89_bert')
    models_dir = '/work01/s-nozue/100knock/chapter09/bert_models'
    model_name = 'bert-base-uncased-classification'

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=10, lr_min=0.00001, warmup_lr_init=0.00001, warmup_t=3, cycle_limit=1)

    train_data_path = '/work01/s-nozue/100knock/chapter06/work/train_data.txt'
    valid_data_path = '/work01/s-nozue/100knock/chapter06/work/valid_data.txt'

    train_dataset = make_dataset_for_collate_fn(train_data_path, tokenizer)
    valid_dataset = make_dataset_for_collate_fn(valid_data_path, tokenizer)

    train_dataloader = make_dataloader(train_dataset, tokenizer, batch_size=16)
    valid_dataloader = make_dataloader(valid_dataset, tokenizer, batch_size=16)

    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            labels = label_filter(labels)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred = torch.argmax(outputs.logits, dim=1)
            correct += (pred == labels).sum().item()
            total += 16

        train_acc = correct / total

        train_loss = total_loss / len(train_dataloader)
        
        acc, eval_loss = eval(model, valid_dataloader)
        wandb.log({'train_loss': train_loss, 'eval_loss': eval_loss, 'train_accuracy': train_acc, 'eval_accuracy': acc, 'lr:': optimizer.param_groups[0]["lr"]})
        print(f'epoch: {epoch + 1}, train_loss: {train_loss}, eval_loss: {eval_loss}, train_accuracy: {train_acc}, eval_accuracy: {acc}, lr: {optimizer.param_groups[0]["lr"]}')
        save_path = f'{models_dir}/{model_name}_{epoch + 1}'

        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        scheduler.step(epoch + 1)

    wandb.finish()
    
if __name__ == '__main__':
    main()