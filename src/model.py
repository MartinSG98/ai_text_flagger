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

def tokenize_data(texts, tokenizer):
    return tokenizer(
        texts,
        padding=True,          # pad short texts to same length
        truncation=True,       # cut long texts to max length
        max_length=512,        # BERT's limit
        return_tensors='pt'    # return PyTorch tensors
    )

def create_dataset(texts, labels, tokenizer):
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512
    )
    
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    })
    
    return dataset

def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        'bert-large-cased',
        num_labels=num_labels
    )
    return model

def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,                  # where to save model checkpoints
        num_train_epochs=3,                     # how many times to loop through training data
        per_device_train_batch_size=16,         # samples processed at once during training
        per_device_eval_batch_size=16,          # samples processed at once during evaluation
        learning_rate=2e-5,                     # step size for weight updates (0.00002)
        weight_decay=0.01,                      # regularization to prevent overfitting
        eval_strategy="epoch",                  # evaluate after each epoch
        save_strategy="epoch",                  # save checkpoint after each epoch
        load_best_model_at_end=True,            # keep best model, not just final one
        logging_dir=f"{output_dir}/logs"        # where to save training logs
    )