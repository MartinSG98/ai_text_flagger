import os
import csv
import json
from docx import Document


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


#debugging purpose - to be deleted
if __name__ == "__main__":
    dataset_path = "Datasets"  # adjust if needed
    files = collect_files(dataset_path)
    print(f"Found {len(files)} files")
    for f in files[:5]:  # print first 5 as sample
        print(f)

def extract_text(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [f.read()]
        
    elif file_path.endswith('.csv'):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            text_column = None
            for name in ['text', 'generated', 'content', 'article']:
                if name in reader.fieldnames:
                    text_column = name
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
                   for key in ['text', 'generated', 'content', 'article']:
                       if key in item:
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