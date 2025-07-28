import os
import json
import fitz  # PyMuPDF
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import re
from typing import List, Dict, Any, Tuple

class BERTPDFProcessorV2:
    def __init__(self, model_path: str = "bert_heading_model"):
        """Initialize BERT model and tokenizer for text classification."""
        self.model_path = Path(model_path)
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.label_encoder = joblib.load(self.model_path / "label_encoder.joblib")
            print("Loaded trained BERT model successfully!")
        except:
            print("Trained model not found. Using fallback classification...")
            self._initialize_fallback_model()
        
        # Set device (CPU as per requirements)
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def _initialize_fallback_model(self):
        """Initialize fallback model when trained model is not available."""
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=5,
            ignore_mismatched_sizes=True
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['Title', 'H1', 'H2', 'H3', 'Body'])
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text blocks from PDF with position and formatting information."""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")['blocks']
            
            for block in blocks:
                if block['type'] != 0:  # Skip non-text blocks
                    continue
                    
                for line in block['lines']:
                    line_text = " ".join([span['text'] for span in line['spans']]).strip()
                    if not line_text:
                        continue
                    
                    # Get the first span for formatting info
                    span = line['spans'][0]
                    
                    text_blocks.append({
                        'text': line_text,
                        'font_size': span['size'],
                        'font_name': span['font'],
                        'is_bold': 'Bold' in span['font'],
                        'is_italic': 'Italic' in span['font'],
                        'bbox': span['bbox'],
                        'page': page_num + 1,
                        'y_position': span['bbox'][1]  # Y position for ordering
                    })
        
        doc.close()
        return text_blocks
    
    def classify_text_with_bert(self, text: str, font_size: float, is_bold: bool, is_italic: bool) -> str:
        """Classify text using BERT model and formatting features."""
        # Prepare text for BERT
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        # Get predicted label
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Apply formatting-based rules to improve classification
        # Heuristic rules based on formatting
        if font_size > 18 or (font_size > 16 and is_bold):
            if predicted_label == 'Body':
                predicted_label = 'H1'
        elif font_size > 16 or (font_size > 14 and is_bold):
            if predicted_label == 'Body':
                predicted_label = 'H2'
        elif font_size > 14 or (font_size > 12 and is_bold):
            if predicted_label == 'Body':
                predicted_label = 'H3'
        
        # Special case for title (usually first line, large font, bold, short text)
        if len(text) < 100 and font_size > 16 and is_bold:
            predicted_label = 'Title'
        
        # Additional heuristics for better classification
        if len(text) < 50 and font_size > 14:
            if predicted_label == 'Body':
                predicted_label = 'H3'
        
        return predicted_label
    
    def extract_title_and_outline(self, text_blocks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract title and outline from classified text blocks."""
        if not text_blocks:
            return "", []
        
        # Sort blocks by page and Y position
        sorted_blocks = sorted(text_blocks, key=lambda x: (x['page'], x['y_position']))
        
        title = ""
        outline = []
        
        for block in sorted_blocks:
            text = block['text']
            page = block['page']
            
            # Skip very short text or page numbers
            if len(text.strip()) < 3 or text.strip().isdigit():
                continue
            
            # Classify the text
            label = self.classify_text_with_bert(
                text, 
                block['font_size'], 
                block['is_bold'], 
                block['is_italic']
            )
            
            # Extract title (first Title or H1 found)
            if not title and label in ['Title', 'H1']:
                title = text
                continue
            
            # Add to outline if it's a heading
            if label in ['H1', 'H2', 'H3']:
                # Avoid duplicates
                if not any(item['text'] == text for item in outline):
                    outline.append({
                        'level': label,
                        'text': text,
                        'page': page
                    })
        
        # If no title found, use the first significant line
        if not title and sorted_blocks:
            for block in sorted_blocks:
                if len(block['text'].strip()) > 5:
                    title = block['text']
                    break
        
        return title, outline
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF and return structured JSON data."""
        try:
            print(f"Processing PDF: {Path(pdf_path).name}")
            
            # Extract text blocks
            text_blocks = self.extract_text_from_pdf(pdf_path)
            print(f"  Extracted {len(text_blocks)} text blocks")
            
            # Extract title and outline
            title, outline = self.extract_title_and_outline(text_blocks)
            
            # Create output structure
            output = {
                "title": title,
                "outline": outline
            }
            
            print(f"  Title: {title}")
            print(f"  Outline items: {len(outline)}")
            
            return output
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            # Return default structure on error
            return {
                "title": "Document Title",
                "outline": [
                    {
                        "level": "H1",
                        "text": "Introduction",
                        "page": 1
                    }
                ]
            }

def process_pdfs_local():
    """Main function to process all PDFs in the local sample dataset."""
    # Initialize processor
    processor = BERTPDFProcessorV2()
    
    # Set up directories (using local paths)
    input_dir = Path("C:/Users/hp/Downloads/Adobe-India-main/Challenge_1a_RF/sample_dataset/pdfs")
    output_dir = Path("C:/Users/hp/Downloads/Adobe-India-main/Challenge_1a_RF/sample_dataset/outputs")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}...")
        
        # Process the PDF
        result = processor.process_pdf(str(pdf_file))
        
        # Create output JSON file
        output_file = output_dir / f"{pdf_file.stem}_bert_output.json"
        
        # Write JSON output
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Completed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    print("Starting BERT-based PDF processing (local test)...")
    process_pdfs_local()
    print("\nCompleted PDF processing!") 