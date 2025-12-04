from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import json
import torch

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

def create_dataset(texts, labels, tokenizer, batch_size=1000):
    all_input_ids = []
    all_attention_masks = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Tokenizing {i}/{len(texts)}...")
        
        tokenized = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=512
        )
        
        all_input_ids.extend(tokenized['input_ids'])
        all_attention_masks.extend(tokenized['attention_mask'])
    
    print(f"Tokenizing {len(texts)}/{len(texts)}... Done!")
    
    dataset = Dataset.from_dict({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
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
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    print("Loading training data...")
    train_texts, train_labels, label_map = load_data('output/train.json')
    print(f"Loaded {len(train_texts)} training samples")
    
    print("Loading validation data...")
    val_texts, val_labels, _ = load_data('output/val.json', label_map)
    print(f"Loaded {len(val_texts)} validation samples")
    
    print("Tokenizing training data...")
    train_dataset = create_dataset(train_texts, train_labels, tokenizer)
    print("Tokenizing validation data...")
    val_dataset = create_dataset(val_texts, val_labels, tokenizer)
    
    print("Loading model...")
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
    
    # Save label map
    with open('output/model/final/label_map.json', 'w') as f:
        json.dump(label_map, f)

    return label_map


def predict(text, model_path='output/model/final'):
    print("Loading tokenizer and model...")
    tokenizer = load_tokenizer()
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Load label map
    with open(f"{model_path}/label_map.json", 'r') as f:
        label_map = json.load(f)
    # Reverse it: {0: "Human", 1: "GPT-4", ...}
    id_to_label = {v: k for k, v in label_map.items()}
    
    print("Tokenizing input text...")
    tokenized = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    print("Making prediction...")
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized)
    
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class_id = torch.max(probabilities, dim=1)
    
    label = id_to_label[predicted_class_id.item()]
    
    # Get all probabilities as dict
    all_probs = {id_to_label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    # Sum all non-Human probabilities
    ai_probability = sum(prob for lbl, prob in all_probs.items() if lbl != "Human")
    
    return {
        "prediction": label,
        "confidence": confidence.item(),
        "ai_probability": ai_probability
    }

if __name__ == "__main__":
    label_map = train()
    print("Training complete!")
    print(f"Label mapping: {label_map}")
