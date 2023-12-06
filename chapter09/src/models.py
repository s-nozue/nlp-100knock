'''
81. RNNによる予測
ID番号で表現された単語列x=(x1,x2,…,xT)がある．
ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
（式は省略）
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import timm
import timm.scheduler
import time

from tokenize_text import tokenize_text

class MyCNN(nn.Module):
    '''
    単語埋め込みの次元数: dw
    畳み込みのフィルターのサイズ: 3 トークン
    畳み込みのストライド: 1 トークン
    畳み込みのパディング: あり
    畳み込み演算後の各時刻のベクトルの次元数: dh
    畳み込み演算後に最大値プーリング（max pooling）を適用し，入力文をdh
    次元の隠れベクトルで表現
    '''

    def __init__(self, emb_weights, drop_ratio):
        super().__init__()
        
        if emb_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=8277)
        else:
            self.embedding = nn.Embedding(8278, 300, padding_idx=8277)

        self.conv = nn.Conv1d(in_channels=300, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50, 4, bias=True)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=x.size()[2])
        x = x.squeeze()
        x = self.fc1(x)
        return x
    
def for_86():
    cnn_net = MyCNN(emb_weights=None)
    softmax = nn.Softmax()
    text = "US STOCKS-Futures drop as Iraq turmoil continues"

    # ランダムに初期化されたパラメータでyを計算してみる
    tokens = tokenize_text('/work01/s-nozue/100knock/chapter09/work/train_data_id.json', text)
    x = torch.tensor(tokens[1], dtype=torch.long)
    x = x.unsqueeze(0)
    print(x)
    y = softmax(cnn_net(x))
    print(y)
    '''
    出力
    tensor([0.1494, 0.2667, 0.3026, 0.2813], grad_fn=<SoftmaxBackward0>)
    '''


class BiRnn(nn.Module):
    '''
    85. 双方向RNN・多層化
    順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．
    '''
    def __init__(self, emb_weights):
        super(BiRnn, self).__init__()

        self.rnn = nn.RNN(300, 50, batch_first=True, num_layers=4, bidirectional=True)
        if emb_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=8277)
        else:
            self.embedding = nn.Embedding(8278, 300, padding_idx=8277)
        self.fc = nn.Linear(100, 4)

    def forward(self, x, length_list):
        # x = pack_padded_sequence(input=x, lengths=length_list, batch_first=True, enforce_sorted=True)
        # x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0, total_length=None)
        x = self.embedding(x)
        x = pack_padded_sequence(input=x, lengths=length_list, batch_first=True, enforce_sorted=True)
        _, x = self.rnn(x, None)
        x1 = x[-2, :, :]
        x2 = x[-1, :, :]
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        return x

class Rnn(nn.Module):
    def __init__(self, emb_weights):
        super(Rnn, self).__init__()

        self.rnn = nn.RNN(300, 50, batch_first=True, num_layers=5)    # default では tanh が使われる
        if emb_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=8277)
        else:
            self.embedding = nn.Embedding(8278, 300, padding_idx=8277)
        self.fc = nn.Linear(50, 4)      # Linear の中にbiasが含まれている

    def forward(self, x, length_list):
        # x = pack_padded_sequence(input=x, lengths=length_list, batch_first=True, enforce_sorted=True)
        # x, _ = pad_packed_sequence(x, batch_first=True, padding_value=8277, total_length=None)
        x = self.embedding(x)
        x = pack_padded_sequence(input=x, lengths=length_list, batch_first=True, enforce_sorted=True)
        _, x = self.rnn(x, None)
        x = x[-1, :, :]
        x = self.fc(x)
        return x

def for_81():
    rnn_net = Rnn()
    softmax = nn.Softmax()
    text = "US STOCKS-Futures drop as Iraq turmoil continues"

    # ランダムに初期化されたパラメータでyを計算してみる
    tokens = tokenize_text('/Users/nozueshinnosuke/workspace/inuilab100knock/100knock-2023/trainee_s-nozue/chapter09/work/train_data_id.json', text)
    x = torch.tensor(tokens[1], dtype=torch.long)
    print(x)
    y = softmax(rnn_net(x))
    print(y)
    
    '''
    出力
    tensor([0.2591, 0.3122, 0.2528, 0.1759], grad_fn=<SoftmaxBackward0>)
    '''


if __name__ == '__main__':
    for_86()