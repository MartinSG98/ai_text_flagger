import json
import random
import re

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