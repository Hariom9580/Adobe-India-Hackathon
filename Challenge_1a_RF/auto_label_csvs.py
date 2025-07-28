import pandas as pd
from pathlib import Path
import re

LABEL_DIR = Path("sample_dataset/labeled")

# Heuristic rules for labeling

def label_row(row, is_first_line):
    text = str(row['text']).strip()
    font_size = row['font_size']
    is_bold = row['is_bold']
    font_name = str(row['font_name'])

    # Title: first large, bold line or first line
    if is_first_line or (font_size > 14 and is_bold):
        if len(text) > 5 and len(text) < 100:
            return 'Title'
    # H1: large, bold, or all caps, or numbered heading
    if font_size > 13 and is_bold:
        return 'H1'
    if re.match(r'^[A-Z][A-Z\s]+$', text) and len(text) > 5:
        return 'H1'
    if re.match(r'^\d+\. ', text):
        return 'H1'
    # H2: medium bold or numbered subheading
    if font_size > 12 and is_bold:
        return 'H2'
    if re.match(r'^\d+\.\d+ ', text):
        return 'H2'
    # H3: slightly larger or bold
    if font_size > 11 and is_bold:
        return 'H3'
    # Body: everything else
    return 'Body'

for csv_file in LABEL_DIR.glob("*_labeled.csv"):
    print(f"Auto-labeling {csv_file.name}...")
    df = pd.read_csv(csv_file)
    labels = []
    for i, row in df.iterrows():
        label = label_row(row, is_first_line=(i==0))
        labels.append(label)
    df['label'] = labels
    df.to_csv(csv_file, index=False)
    print(f"Labeled: {csv_file}")
print("Auto-labeling complete! You can now use these for ML training.") 