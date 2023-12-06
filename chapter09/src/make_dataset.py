import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorForTokenClassification, default_data_collator

def tokenize_text(text, tokenizer):
    result = tokenizer(text, return_tensors='pt')

    return {
        'input_ids': result['input_ids'][0],
        'attention_mask': result['attention_mask'][0],
    }

def make_dataset_for_collate_fn(dataset_path, tokenizer):
    '''
    pandas categotical を使ってカテゴリを数値に変換する
    e: 0, t: 1, m: 2, b: 3
    '''
    dataset = []

    df = pd.read_csv(dataset_path, sep='\t', names=['inputs', 'labels'])

    # category を数値に変換
    df['labels'] = pd.Categorical(df['labels'])
    df['labels'] = df['labels'].cat.codes

    for data in df.itertuples():
        text = data[1]
        label = data[2]
        tokenized_text = tokenize_text(text, tokenizer)
        tokenized_text['labels'] = torch.tensor(label, dtype=torch.long).unsqueeze(0)
        dataset.append(tokenized_text)

    return dataset


def make_dataloader(dataset, tokenizer, batch_size):
    '''
    dataset を dataloader に変換する
    '''

    # data_collator = default_data_collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        pad_to_multiple_of=8, 
        padding=True, 
        return_tensors='pt', 
        max_length=64
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    return dataloader

def check_max_length(dataset, tokenizer):
    '''
    dataset の最大系列長を確認する
    '''
    max_length = 0
    for data in dataset:
        text = data[0]
        tokenized_text = tokenizer(text)
        max_length = max(max_length, len(tokenized_text['input_ids']))

    return max_length

def main():
    dataset_path = '/Users/nozueshinnosuke/workspace/inuilab100knock/100knock-2023/trainee_s-nozue/chapter06/work/test_data.txt'

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = make_dataset_for_collate_fn(dataset_path, tokenizer)
    # print(check_max_length(dataset, tokenizer))
    # print(dataset)

    dataloader = make_dataloader(dataset, tokenizer, batch_size=8)

    return dataloader

if __name__ == '__main__':
    main()