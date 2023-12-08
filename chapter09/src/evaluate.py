import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

from make_dataset import make_dataset_for_collate_fn, make_dataloader
from train_bert import eval

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'generate'])
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--input_file', type=str, default='/work01/s-nozue/100knock/chapter06/work/test_data.txt')
    parser.add_argument('--input_text', type=str)
    args = parser.parse_args()
    return args.mode, args.model_name_or_path, args.input_file, args.input_text

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode, model_name_or_path, input_file, input_text = parser()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=4)
    model.to(device)

    if mode == 'eval':
        dataset = make_dataset_for_collate_fn(input_file, tokenizer)
        dataloader = make_dataloader(dataset, tokenizer, batch_size=8)

        acc, eval_loss = eval(model, dataloader)

        print(f'accuracy: {acc}, eval_loss: {eval_loss}')
        df = pd.DataFrame({'model_name_or_path': [os.path.basename(model_name_or_path)], 'accuracy': [acc], 'eval_loss': [eval_loss]})
        df.to_csv('/work01/s-nozue/100knock/chapter09/eval_result.csv', mode='a', header=False, index=False, sep='\t', float_format='%.3f')
    
    elif mode == 'generate':
        categories = {0: 'Entertainment', 1: 'Technology', 2: 'Medical', 3: 'Business'}
        
        with torch.no_grad():
            model.eval()
            inputs = tokenizer(input_text, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1)
            print(categories[pred.item()])
            df = pd.DataFrame({'model_name_or_path': [os.path.basename(model_name_or_path)], 'category': [categories[pred.item()]]})
            df.to_csv('/work01/s-nozue/100knock/chapter09/generate_result.csv', mode='a', header=False, index=False, sep='\t')

if __name__ == '__main__':
    main()