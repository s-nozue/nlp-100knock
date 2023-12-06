import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import timm
import timm.scheduler
import time
import pandas as pd
import wandb
import json
from gensim.models import KeyedVectors

from tokenize_text import tokenize_text
from models import MyCNN
from train_rnn import cal_cross_entropy_loss, collate_fn, make_weights_matrix

def eval(model, X_eval, Y_eval, batch_size=1):
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        for X, y in zip(X_eval, Y_eval):
            X = X.to('cuda')
            y = y.to('cuda')
            output = model(X)
            eval_loss += cal_cross_entropy_loss(output, y)
            total += batch_size
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
        eval_loss = eval_loss / len(X_eval)
        acc = correct / total
        return eval_loss, acc

def data_loader(x, y, batch_size):
    x_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, collate_fn=collate_fn)    
    y_loader = torch.utils.data.DataLoader(y, batch_size=batch_size)

    return x_loader, y_loader

def tokenize_dataset(dataset, file):
    X_tokenized = []
    
    for i in range(len(dataset)):
        _, X_tokenized_i = tokenize_text(file, dataset[i])
        X_tokenized_i = torch.tensor(X_tokenized_i, dtype=torch.long)
        X_tokenized.append(X_tokenized_i)

    return X_tokenized

def main(mode):
    '''
    87. 確率的勾配降下法によるCNNの学習
    確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題86で構築したモデルを学習せよ．
    訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．

    88. パラメータチューニング
    問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，高性能なカテゴリ分類器を構築せよ
    '''
    start = time.time()

    ids_file = '/work01/s-nozue/100knock/chapter09/work/train_data_id.json'
    weights_matrix = make_weights_matrix(ids_file)

    if mode == 'sweep':
        wandb.init()
        drop_ratio = wandb.run.config["drop_ratio"]
    else:
        drop_ratio = 0.0

    # モデルの定義
    cnn_net = MyCNN(weights_matrix, drop_ratio)
    cnn_net = cnn_net.to('cuda')

    if mode == 'train':
        wandb.init(project='chapter09_87')
        learning_rate = 0.1
        weight_decay = 0.0
        warmup_t = 3
        optimizer = optim.SGD(cnn_net.parameters(), lr=0.1)
        name = f'cnn-lr{learning_rate}-wd{weight_decay}-wr{warmup_t}-drop0.3-batch16-epoch10'

    elif mode == 'sweep':
        name = f'cnn-lr{wandb.run.config["learning_rate"]}-wd{wandb.run.config["weight_decay"]}-wr{wandb.run.config["warmup_t"]}-drop{wandb.run.config["drop_ratio"]}-batch16-epoch10'
        wandb.run.name = name
        print(wandb)
        print("wandb run name:", wandb.run.name)
        print("wandb run id:", wandb.run.id)
        print("wandb run path:", wandb.run.config)
        learning_rate = wandb.run.config['learning_rate']
        weight_decay = wandb.run.config['weight_decay']
        warmup_t = wandb.run.config['warmup_t']
        optimizer = optim.AdamW(cnn_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=10, lr_min=0.0001, warmup_lr_init=0.0001, warmup_t=warmup_t, cycle_limit=1)

    # データの読み込み
    train_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/train_data.txt', sep='\t', header=None)[0]
    train_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/train_data_Y.pt')

    valid_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/valid_data.txt', sep='\t', header=None)[0]
    valid_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/valid_data_Y.pt')

    # データの前処理
    train_X = tokenize_dataset(train_X, ids_file)
    valid_X = tokenize_dataset(valid_X, ids_file)

    # train data のミニバッチ化
    batch_size = 16
    train_X, train_Y = data_loader(train_X, train_Y, batch_size)
    # valid data のpadding
    valid_X, valid_Y = data_loader(valid_X, valid_Y, batch_size)

    # 学習
    for epoch in range(10):
        correct = 0
        total = 0
        cnn_net.train()
        for X, y in zip(train_X, train_Y):
            X = X.to('cuda')
            y = y.to('cuda')
            optimizer.zero_grad()
            output = cnn_net(X,)
            pred = torch.argmax(output, dim=1)
            loss = cal_cross_entropy_loss(output, y)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
            total += batch_size

        train_acc = correct / total

        eval_loss, eval_acc = eval(cnn_net, valid_X, valid_Y, batch_size)
        wandb.log({'train_loss': loss.item(), 'train_accuracy': train_acc, 'eval_loss': eval_loss.item(), 'eval_accuracy': eval_acc, 'lr:': optimizer.param_groups[0]["lr"]})
        print(f'epoch: {epoch + 1}, train_loss: {loss.item()}, train_accuracy: {train_acc}, eval_loss: {eval_loss}, eval_accuracy: {eval_acc}')
        scheduler.step(epoch + 1)
    
    end = time.time()
    print(end-start)
    wandb.finish()

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sweep'])
    args = parser.parse_args()
    return args.mode

if __name__ == "__main__":
    mode = main_parser()

    if mode == 'train':
        main(mode)

    elif mode == 'sweep':
        sweep_configuration = {
            'method': 'grid',
            'metric': {
                    'name': 'eval_loss',
                    'goal': 'minimize'
                },

                'parameters':{
                    'learning_rate': {
                        'values': [1e-3],
                    },

                    'weight_decay': {
                        'values': [0.001],
                    },

                    'warmup_t': {
                        'values': [3],
                    },
                    'drop_ratio': {
                        'values': [0.1, 0.2, 0.3, 0.5]
                    }
                }
            }

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, 
            project='chapter09_88_sweep_drop',
            )
        wandb.agent(sweep_id, function=lambda: main(mode))
    
    else:
        raise ValueError(f'--mode {mode} is invalid.')