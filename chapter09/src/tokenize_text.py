'''
80. ID番号への変換
問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，学習データ中で2回以上出現する単語にID番号を付与せよ．
そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．
'''

import os
import re
import argparse
import json
import pathlib

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['make_id', 'tokenize'])
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--text', help='Text you want to tokenize.')

    args = parser.parse_args()
    return args.mode, args.input_file, args.output_file, args.text

def get_words(train_data):
    with open(train_data, 'r') as f:
        words = []
        for line in f:
            line = line.rstrip()
            line = line.split()
            words.extend(line[:-1])
    return words

def preprocessing(words):
    words_removed_signal = remove_signal(words)
    words_removed_signal = [w for w in words_removed_signal if w != '']
    return [word.lower() for word in words_removed_signal]

def remove_signal(words):
    signals = ''.join(['.', ',', ':', ';', '!', '?', '(', ')', '[', ']', "'", '"', '/'])
    words_removed_signal = []

    for word in words:
        word = word.rstrip(signals)
        word = word.lstrip(signals)
        words_removed_signal.append(word)
    
    return words_removed_signal

def numbering_words(words_removed_signal):
    '''
    最も頻出する単語から順にIDを付与する
    '''
    words_dict = {}
    for word in words_removed_signal:
        if word in words_dict:
            words_dict[word] += 1
        else:
            words_dict[word] = 1

    words_dict_sorted = sorted(words_dict.items(), key=lambda x:x[1], reverse=True)
    # 2回未満の単語のID番号はすべて0として、ID番号を付与してdictに格納する
    words_dict_numbered = {k:0 if v < 2 else i+1 for i, (k, v) in enumerate(words_dict_sorted)}
    return words_dict_numbered

def output_id(words_dict_numbered, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        file = pathlib.Path(output_file)
        file.touch()
    with open(output_file, 'w') as f:
        json.dump(words_dict_numbered, f, ensure_ascii=False)

def make_id(train_data, output_file):
    words = get_words(train_data)
    words_removed_signal = preprocessing(words)
    words_dict_numbered = numbering_words(words_removed_signal)
    output_id(words_dict_numbered, output_file)


# 以下、tokenizeの関数
def ids_for_tokenizing(input_file):
    with open(input_file, 'r') as f:
        words_dict_numbered = json.load(f)
    return words_dict_numbered

def output_tokenized_text(text_preprocessed, text_tokenized, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        file = pathlib.Path(output_file)
        file.touch()
    
    output_text = f'{text_preprocessed} -> {text_tokenized}\n'

    with open(output_file, 'w') as f:
        f.write(output_text)

def tokenize_text(input_file, text):
    words_dict_numbered = ids_for_tokenizing(input_file)
    text = text.split()
    text_preprocessed = preprocessing(text)
    text_tokenized = [words_dict_numbered[word] if word in words_dict_numbered else 0 for word in text_preprocessed]
    return text_preprocessed, text_tokenized

if __name__ == '__main__':
    mode, input_file, output_file, text = argparser()

    if mode == 'make_id':
        make_id(input_file, output_file)

    elif mode == 'tokenize':
        text_preprocessed, tokenized_text = tokenize_text(input_file, text)
        output_tokenized_text(text_preprocessed, tokenized_text, output_file)
        print(text_preprocessed, tokenized_text)

    else:
        raise ValueError('mode is invalid')