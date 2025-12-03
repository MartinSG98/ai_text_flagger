from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import json

def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-large-cased')

def load_data(path, label_map=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    string_labels = [item['label'] for item in data]
    
    # Build mapping only if not provided
    if label_map is None:
        unique_labels = sorted(set(string_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert string labels to integers
    labels = [label_map[label] for label in string_labels]
    
    return texts, labels, label_map

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
    return BertForSequenceClassification.from_pretrained(
        'bert-large-cased',
        num_labels=num_labels
    )

def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,                  # where to save model checkpoints
        num_train_epochs=10,                    # how many times to loop through training data
        per_device_train_batch_size=16,         # samples processed at once during training
        per_device_eval_batch_size=16,          # samples processed at once during evaluation
        learning_rate=2e-5,                     # step size for weight updates (0.00002)
        weight_decay=0.01,                      # regularization to prevent overfitting
        eval_strategy="epoch",                  # evaluate after each epoch
        save_strategy="epoch",                  # save checkpoint after each epoch
        load_best_model_at_end=True,            # keep best model, not just final one
        logging_dir=f"{output_dir}/logs"        # where to save training logs
    )

def train():
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Load data
    train_texts, train_labels, label_map = load_data('output/train.json')
    val_texts, val_labels, _ = load_data('output/val.json', label_map)
    
    # Create datasets
    train_dataset = create_dataset(train_texts, train_labels, tokenizer)
    val_dataset = create_dataset(val_texts, val_labels, tokenizer)
    
    # Load model
    model = load_model(num_labels=len(label_map))
    
    # Training args
    training_args = get_training_args('output/model')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train!
    trainer.train()
    
    # Save
    trainer.save_model('output/model/final')
    
    return label_map

if __name__ == "__main__":
    label_map = train()
    print("Training complete!")
    print(f"Label mapping: {label_map}")
