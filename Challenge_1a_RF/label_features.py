import pandas as pd
import json
from pathlib import Path
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def label_features(features_csv, outline_json, output_csv, threshold=0.85):
    df = pd.read_csv(features_csv)
    with open(outline_json, 'r', encoding='utf-8') as f:
        outline = json.load(f)
    outline_entries = outline.get('outline', [])
    # Build a list of (level, text, page)
    outline_tuples = [
        (entry['level'], entry['text'].strip().lower(), entry['page'])
        for entry in outline_entries
    ]
    labels = []
    for _, row in df.iterrows():
        text = row['text'].strip().lower()
        page = row['page']
        label = 'Other'
        for level, otext, opage in outline_tuples:
            if page == opage and (text == otext or similar(text, otext) > threshold):
                label = level
                break
        labels.append(label)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"Labeled features saved to {output_csv}")

def label_all_features(features_dir, outputs_dir, labeled_dir):
    features_dir = Path(features_dir)
    outputs_dir = Path(outputs_dir)
    labeled_dir = Path(labeled_dir)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    for features_csv in features_dir.glob('*.csv'):
        base = features_csv.stem
        json_file = outputs_dir / f"{base}.json"
        if json_file.exists():
            output_csv = labeled_dir / f"{base}_labeled.csv"
            label_features(features_csv, json_file, output_csv)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        label_all_features(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python label_features.py <features_dir> <outputs_dir> <labeled_dir>") 