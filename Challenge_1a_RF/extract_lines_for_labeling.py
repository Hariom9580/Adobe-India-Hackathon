import os
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

PDF_DIR = Path("sample_dataset/pdfs")
OUT_DIR = Path("sample_dataset/labeled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_file in PDF_DIR.glob("*.pdf"):
    print(f"Extracting lines from {pdf_file.name}...")
    doc = fitz.open(pdf_file)
    rows = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                line_text = " ".join([span['text'] for span in line['spans']]).strip()
                if not line_text:
                    continue
                span = line['spans'][0]
                rows.append({
                    'text': line_text,
                    'font_size': span['size'],
                    'font_name': span['font'],
                    'is_bold': int('Bold' in span['font']),
                    'is_italic': int('Italic' in span['font']),
                    'x0': span['bbox'][0],
                    'y0': span['bbox'][1],
                    'x1': span['bbox'][2],
                    'y1': span['bbox'][3],
                    'page': page_num,
                    'label': ''  # To be filled in manually
                })
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{pdf_file.stem}_labeled.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
print("Extraction complete! Now label the CSVs in sample_dataset/labeled/.") 