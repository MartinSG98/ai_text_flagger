import json
import random
import re
import os

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces/newlines with single space
    return text.strip()

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(dataset)
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]
    
    return train_set, val_set, test_set

def preprocess_dataset(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    cleaned_dataset = [{"text": clean_text(item["text"]), "label": item["label"], "source_file": item["source_file"]} for item in dataset]
    train_set, val_set, test_set = split_dataset(cleaned_dataset)
    #save splits
    with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_set, f)
    with open(os.path.join(output_dir, "val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_set, f)
    with open(os.path.join(output_dir, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_set, f)

#used for debugging - to be deleted
if __name__ == "__main__":
    preprocess_dataset("output/dataset.json", "output")
    print("Done!")