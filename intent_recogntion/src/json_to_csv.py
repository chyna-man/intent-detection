import os
import json
import pandas as pd


def parse_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for intent, samples in data.items():
        for sample in samples:
            text = ''.join([chunk['text'] for chunk in sample['data']])
            rows.append({"text": text, "label": intent})
    return rows


def convert_directory_to_csv(input_dir, output_csv):
    all_rows = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            rows = parse_json_file(file_path)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} samples to {output_csv}")

convert_directory_to_csv("datasets/Train", "train.csv")
convert_directory_to_csv("datasets/Validate", "valid.csv")
