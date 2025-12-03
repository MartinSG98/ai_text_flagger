import os
import csv
import json
from docx import Document
import sys

csv.field_size_limit(100_000_000)  # 100 million characters

def collect_files(dataset_path):
    files = []
    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(('.txt', '.csv', '.json', '.docx')):
                if "Human written text" in root:
                    label = "Human"
                elif "ai generated text" in root.lower():
                    label = os.path.basename(root)
                else:
                    continue  # skip files not in labeled directories
                files.append((os.path.join(root, filename), label))
    return files


def extract_text(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [f.read()]
        
    elif file_path.endswith('.csv'):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            text_column = None
            for field in reader.fieldnames:
                if field.lower() in ['text', 'generated', 'content', 'article']:
                    text_column = field
                    break
            if text_column is None:
                print(f"No suitable text column found in {file_path}")
                return []
            for row in reader:
                texts.append(row[text_column]) 
        return texts
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = []
                for item in data:
                    for key in item.keys():
                        if key.lower() in ['text', 'generated', 'content', 'article']:
                            texts.append(item[key])
                            break
                return texts
            else:
                return []
            
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        texts = [para.text for para in doc.paragraphs]
        return ['\n'.join(texts)]
    
    else:
        return []  # unsupported file type
    

def build_dataset(dataset_path, output_path):
    dataset = []
    files = collect_files(dataset_path)
    for file_path, label in files:
        print(f"Processing: {file_path}")  # add this line
        texts = extract_text(file_path)
        for text in texts:
            dataset.append({"text": text,"label": label, "source_file": file_path})
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    return dataset
    

    #debugging purpose - to be deleted
if __name__ == "__main__":
    build_dataset("Datasets", "output/dataset.json")
    print("Done!")