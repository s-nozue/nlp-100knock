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
from models import Rnn

def cal_cross_entropy_loss(y_hat, y):
    # print(y_hat)
    # print(y_hat.size())
    # print(y)
    criterion = nn.CrossEntropyLoss()
    return criterion(y_hat, y)

def cal_gradient(y_hat, y, x):
    loss = cal_cross_entropy_loss(y_hat, y)
    print('loss:', loss)
    loss.backward()
    return x.grad

def eval(model, X_eval, Y_eval, lengths, batch_size=1):
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        total = 0
        correct = 0
        for X, y, length in zip(X_eval, Y_eval, lengths):
            X, y, length = sort_by_length(X, y, length)
            X = X.to('cuda')
            y = y.to('cuda')
            output = model(X, length)
            eval_loss += cal_cross_entropy_loss(output, y)
            pred = torch.argmax(output, dim=1)
            total += batch_size
            correct += (pred == y).sum().item()
        eval_loss = eval_loss / len(X_eval)
        acc = correct / total
        return eval_loss, acc

def tokenized_dataset(dataset, file):
    X_tokenized = []
    lengths = []
    
    for i in range(len(dataset)):
        _, X_tokenized_i = tokenize_text(file, dataset[i])
        X_tokenized_i = torch.tensor(X_tokenized_i, dtype=torch.long)
        X_tokenized.append(X_tokenized_i)
        lengths.append(len(X_tokenized_i))

    return X_tokenized, lengths

def data_loader(x, y, lengths, batch_size):
    x_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, collate_fn=collate_fn)    
    y_loader = torch.utils.data.DataLoader(y, batch_size=batch_size)
    length_loader = torch.utils.data.DataLoader(lengths, batch_size=batch_size)
    return x_loader, y_loader, length_loader

def collate_fn(batch):
    '''
    paddingして系列長を揃える
    '''
    x = []
    for b in batch:
        x.append(torch.tensor(b, dtype=torch.long))
    x = pad_sequence(x, batch_first=True, padding_value=8277)
    return x

def sort_by_length(x_batch, y_batch, lengths):
    '''
    系列長でソートする
    '''
    lengths, sorted_idx = lengths.sort(descending=True)
    x_batch = x_batch[sorted_idx]
    y_batch = y_batch[sorted_idx]
    return x_batch, y_batch, lengths

def make_weights_matrix(ids_file):
    '''
    Google Newsデータセット（約1,000億単語）での学習済み単語ベクトルをidの単語に対応させる
    '''
    with open(ids_file, 'r') as f:
        json_data = json.load(f)

    word2vec = KeyedVectors.load_word2vec_format('/work01/s-nozue/100knock/chapter07/data/GoogleNews-vectors-negative300.bin', binary=True)
    weights_matrix = torch.zeros(8278, 300)

    for i in json_data.values():
        try:
            weights_matrix[i] = torch.tensor(word2vec[json_data[i]], dtype=torch.float32)
        except:
            # ない場合はランダムな数で埋める
            weights_matrix[i] = torch.randn(300, dtype=torch.float32, requires_grad=True)
    
    return weights_matrix


def for_84():
    '''
    84. 単語ベクトルの導入
    事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)
    を初期化し，学習せよ．
    '''

    wandb.init(project='chapter09_84')
    ids_file = '/work01/s-nozue/100knock/chapter09/work/train_data_id.json'
    weights_matrix = make_weights_matrix(ids_file)

    rnn_net = Rnn(emb_weights=weights_matrix)
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


def for_83():
    '''
    83. ミニバッチ化・GPU上での学習
    問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．
    また，GPU上で学習を実行せよ
    '''

    wandb.init(project='chapter09_83')
    rnn_net = Rnn()
    rnn_net.to('cuda')
    optimizer = optim.SGD(rnn_net.parameters(), lr=0.1)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=10, lr_min=0.0001, warmup_lr_init=0.0001, warmup_t=3, cycle_limit=1)

    # データの読み込み
    train_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/train_data.txt', sep='\t', header=None)[0]
    train_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/train_data_Y.pt')

    valid_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/valid_data.txt', sep='\t', header=None)[0]
    valid_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/valid_data_Y.pt')

    # データの前処理
    train_X, train_lengths = tokenized_dataset(train_X)
    valid_X, valid_lengths = tokenized_dataset(valid_X)

    # train data のミニバッチ化
    train_X, train_Y, train_lengths = data_loader(train_X, train_Y, train_lengths, 8)
    # valid data のpadding
    valid_X, valid_Y, valid_lengths = data_loader(valid_X, valid_Y, valid_lengths, 8)

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

        eval_loss, acc = eval(rnn_net, valid_X, valid_Y, valid_lengths, 8)
        wandb.log({'train_loss': loss.item(), 'eval_loss': eval_loss.item(), 'accuracy': acc, 'lr:': optimizer.param_groups[0]["lr"]})
        print(f'epoch: {epoch + 1}, train_loss: {loss.item()}, eval_loss: {eval_loss}, accuracy: {acc}')
        scheduler.step(epoch + 1)

def for_82():
    '''
    82. 確率的勾配降下法による学習
    確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．
    訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ
    '''
    wandb.init(project='chapter09')
    rnn_net = Rnn()
    rnn_net.to('cuda')
    optimizer = optim.SGD(rnn_net.parameters(), lr=0.2)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=10, lr_min=0.0001, warmup_lr_init=0.0001, warmup_t=3, cycle_limit=1)

    # データの読み込み
    train_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/train_data.txt', sep='\t', header=None)[0]
    train_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/train_data_Y.pt')

    valid_X = pd.read_csv('/work01/s-nozue/100knock/chapter06/work/valid_data.txt', sep='\t', header=None)[0]
    valid_Y = torch.load('/work01/s-nozue/100knock/chapter08/datasets/valid_data_Y.pt')

    # データの前処理
    train_X = tokenized_dataset(train_X)
    valid_X = tokenized_dataset(valid_X)

    # train data のミニバッチ化
    train_X, train_Y = data_loader(train_X, train_Y, 1)
    # valid data のpadding
    valid_X, valid_Y = data_loader(valid_X, valid_Y, 1)

    # 学習
    for epoch in range(10):
        rnn_net.train()
        for X, y in zip(train_X, train_Y):
            X = X.to('cuda')
            y = y.to('cuda')
            print(len(X), len(y))
            optimizer.zero_grad()
            output = rnn_net(X)
            # print(output.shape(), y.shape())
            loss = cal_cross_entropy_loss(output, y)
            loss.backward()
            optimizer.step()

        eval_loss, acc = eval(rnn_net, valid_X, valid_Y)
        wandb.log({'train_loss': loss.item(), 'eval_loss': eval_loss.item(), 'accuracy': acc, 'lr:': optimizer.param_groups[0]["lr"]})
        print(f'epoch: {epoch + 1}, train_loss: {loss.item()}, eval_loss: {eval_loss}, accuracy: {acc}')
        scheduler.step(epoch + 1)

if __name__ == '__main__':
    # for_83()
    # for_82()
    for_84()