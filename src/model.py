from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json

def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-large-cased')

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    string_labels = [item['label'] for item in data]
    
    # Build mapping from unique labels
    unique_labels = sorted(set(string_labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert string labels to integers
    labels = [label_map[label] for label in string_labels]
    
    return texts, labels, label_map