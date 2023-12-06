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
from models import BiRnn
from train_rnn import cal_cross_entropy_loss, eval, tokenized_dataset, data_loader, sort_by_length, collate_fn, make_weights_matrix

def for_85():
    '''
    84. 単語ベクトルの導入
    事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)
    を初期化し，学習せよ．
    '''

    wandb.init(project='chapter09_85')
    ids_file = '/work01/s-nozue/100knock/chapter09/work/train_data_id.json'
    weights_matrix = make_weights_matrix(ids_file)

    rnn_net = BiRnn(emb_weights=weights_matrix)
    rnn_net.to('cuda')
    optimizer = optim.SGD(rnn_net.parameters(), lr=0.1)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=10, lr_min=0.0001, warmup_lr_init=0.0001, warmup_t=3, cycle_limit=1)

    # データの読み込み
    train_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/train_data.txt', sep='\t', header=None)[0]
    train_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/train_data_Y.pt')

    valid_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/valid_data.txt', sep='\t', header=None)[0]
    valid_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/valid_data_Y.pt')

    # データの前処理
    train_X, train_lengths = tokenized_dataset(train_X, ids_file)
    valid_X, valid_lengths = tokenized_dataset(valid_X, ids_file)

    # train data のミニバッチ化
    train_X, train_Y, train_lengths = data_loader(train_X, train_Y, train_lengths, 16)
    # valid data のpadding
    valid_X, valid_Y, valid_lengths = data_loader(valid_X, valid_Y, valid_lengths, 16)

    # 学習
    for epoch in range(10):
        rnn_net.train()
        for X, y, length in zip(train_X, train_Y, train_lengths):
            X, y, length = sort_by_length(X, y, length)
            X = X.to('cuda')
            y = y.to('cuda')
            # print(X)
            optimizer.zero_grad()
            output = rnn_net(X, length)
            loss = cal_cross_entropy_loss(output, y)
            loss.backward()
            optimizer.step()

        eval_loss, acc = eval(rnn_net, valid_X, valid_Y, valid_lengths, 16)
        wandb.log({'train_loss': loss.item(), 'eval_loss': eval_loss.item(), 'accuracy': acc, 'lr:': optimizer.param_groups[0]["lr"]})
        print(f'epoch: {epoch + 1}, train_loss: {loss.item()}, eval_loss: {eval_loss}, accuracy: {acc}')
        scheduler.step(epoch + 1)

if __name__ == '__main__':
    for_85()