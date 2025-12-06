import json
import random

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    text = ' '.join(text.split())
    return text.strip()

def convert_to_binary(label):
    #Convert all non-Human labels to 'AI'
    if label == "Human":
        return "Human"
    return "AI"

def split_dataset(data, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

def preprocess_dataset(input_path, output_dir):
    data = load_dataset(input_path)
    
    # Clean text and convert to binary labels
    for item in data:
        item['text'] = clean_text(item['text'])
        item['label'] = convert_to_binary(item['label'])
    
    train, val, test = split_dataset(data)
    
    with open(f'{output_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f)
    with open(f'{output_dir}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f)
    with open(f'{output_dir}/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    preprocess_dataset("output/dataset.json", "output")
    print("Done!")