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