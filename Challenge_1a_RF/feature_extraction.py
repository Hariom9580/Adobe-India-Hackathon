import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    features = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if block['type'] != 0:
                continue  # skip images, etc.
            for line in block['lines']:
                line_text = " ".join([span['text'] for span in line['spans']]).strip()
                if not line_text:
                    continue
                # Use the first span as representative for font features
                span = line['spans'][0]
                features.append({
                    'text': line_text,
                    'font_size': span['size'],
                    'font_name': span['font'],
                    'is_bold': int('Bold' in span['font']),
                    'is_italic': int('Italic' in span['font']),
                    'x0': span['bbox'][0],
                    'y0': span['bbox'][1],
                    'x1': span['bbox'][2],
                    'y1': span['bbox'][3],
                    'page': page_num
                })
    return pd.DataFrame(features)

def extract_features_from_dir(pdf_dir, output_dir):
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_dir.glob("*.pdf"):
        output_csv = output_dir / f"{pdf_file.stem}.csv"
        df = extract_features_from_pdf(pdf_file)
        df.to_csv(output_csv, index=False)
        print(f"Extracted features from {pdf_file} to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        pdf_path = sys.argv[1]
        output_csv = sys.argv[2]
        if pdf_path.lower().endswith('.pdf'):
            df = extract_features_from_pdf(pdf_path)
            df.to_csv(output_csv, index=False)
            print(f"Extracted features to {output_csv}")
        else:
            # Assume directory mode
            extract_features_from_dir(pdf_path, output_csv)
            print(f"Extracted features for all PDFs in {pdf_path} to {output_csv}")
    else:
        print("Usage:")
        print("  python feature_extraction.py <pdf_path> <output_csv>")
        print("  python feature_extraction.py <pdf_dir> <output_dir>") 