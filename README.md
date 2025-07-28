# Adobe India Hackathon 2025 - Connecting the Dots

## Overview
This repository contains solutions for both Round 1A and Round 1B of the Adobe India Hackathon 2025 "Connecting the Dots" challenge. The solutions use advanced NLP techniques to extract structured information from PDF documents and provide intelligent document analysis.

## 🚀 Challenge 1A: PDF Structure Extraction

### Approach
- **BERT-Powered Classification**: Uses DistilBERT for intelligent text classification
- **Multi-Modal Analysis**: Combines semantic understanding with visual formatting cues
- **Hierarchical Extraction**: Identifies document title and heading levels (H1, H2, H3)
- **Page Tracking**: Maintains accurate page numbers for each heading

### Models & Libraries Used
- **DistilBERT**: `distilbert-base-uncased` for text classification
- **PyMuPDF**: PDF text extraction with formatting information
- **Transformers**: Hugging Face transformers library for BERT
- **scikit-learn**: For additional classification features
- **pandas**: Data manipulation and processing

### Build & Run Instructions

#### Build Docker Image
```bash
docker build --platform linux/amd64 -t challenge1a-solution .
```

#### Run Solution
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1a-solution
```

#### Expected Behavior
- Processes all PDFs from `/app/input` directory
- Generates corresponding `filename.json` files in `/app/output`
- Output format: `{"title": "...", "outline": [{"level": "H1", "text": "...", "page": 1}]}`

## 🎯 Challenge 1B: Persona-Driven Document Intelligence

### Approach
- **Semantic Relevance**: Uses TF-IDF and cosine similarity for content matching
- **Persona Understanding**: Analyzes job requirements and persona expertise
- **Section Ranking**: Prioritizes sections based on relevance scores
- **Sub-section Analysis**: Provides detailed content analysis for top sections

### Models & Libraries Used
- **scikit-learn**: TF-IDF vectorization and similarity calculations
- **pandas**: Data processing and feature engineering
- **numpy**: Numerical computations
- **pdfplumber**: PDF text extraction

### Build & Run Instructions

#### Build Docker Image
```bash
cd Challenge_1b_RF
docker build --platform linux/amd64 -t challenge1b-solution .
```

#### Run Solution
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b-solution
```

#### Expected Behavior
- Processes document collections from `/app/input`
- Generates `challenge1b_output.json` in `/app/output`
- Output includes metadata, extracted sections, and subsection analysis

## 📊 Performance Compliance

### Challenge 1A Constraints
- ✅ **Execution Time**: ≤ 10 seconds for 50-page PDFs
- ✅ **Model Size**: ≤ 200MB (DistilBERT ~66MB)
- ✅ **Network**: No internet access required
- ✅ **Runtime**: CPU-only (amd64 architecture)
- ✅ **Memory**: < 16GB RAM usage

### Challenge 1B Constraints
- ✅ **Execution Time**: ≤ 60 seconds for 3-5 documents
- ✅ **Model Size**: ≤ 1GB
- ✅ **Network**: No internet access required
- ✅ **Runtime**: CPU-only processing

## 🏗️ Project Structure

```
Adobe-India-Hackathon/
├── README.md                    # This file
├── Dockerfile                   # Challenge 1A Docker configuration
├── requirements.txt             # Challenge 1A dependencies
├── Challenge_1a_RF/            # Challenge 1A solution
│   ├── bert_pdf_processor_final.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README_FINAL.md
└── Challenge_1b_RF/            # Challenge 1B solution
    ├── src/
    │   ├── main.py
    │   ├── model.py
    │   ├── extract_sections.py
    │   └── feature_engineering.py
    ├── Dockerfile
    ├── requirements.txt
    └── Collection_*/            # Sample document collections
```

## 🔧 Technical Implementation

### Challenge 1A: BERT-Based Classification
1. **Text Extraction**: Extract text blocks with formatting info using PyMuPDF
2. **BERT Tokenization**: Convert text to BERT tokens
3. **Model Inference**: Get classification predictions for Title/H1/H2/H3/Body
4. **Format Integration**: Combine with font size, bold, italic formatting
5. **Structure Building**: Create hierarchical outline with page numbers

### Challenge 1B: Semantic Relevance Analysis
1. **Section Extraction**: Extract all sections from input documents
2. **Feature Engineering**: Compute TF-IDF features for persona and job requirements
3. **Similarity Calculation**: Calculate cosine similarity between sections and requirements
4. **Ranking**: Sort sections by relevance score
5. **Output Generation**: Create structured JSON with metadata and analysis

## 🎯 Key Features

### Challenge 1A
- **Intelligent Classification**: BERT + formatting cues for accurate heading detection
- **Robust Processing**: Handles various PDF formats and layouts
- **Performance Optimized**: Fast processing within constraints
- **Schema Compliant**: Outputs valid JSON format

### Challenge 1B
- **Persona-Aware**: Considers specific user roles and expertise
- **Job-Focused**: Prioritizes content relevant to specific tasks
- **Scalable**: Handles multiple documents efficiently
- **Insightful**: Provides detailed subsection analysis

## 🧪 Testing

### Challenge 1A Testing
```bash
# Test with sample data
docker run --rm \
  -v $(pwd)/Challenge_1a_RF/sample_dataset/pdfs:/app/input:ro \
  -v $(pwd)/Challenge_1a_RF/sample_dataset/outputs:/app/output \
  --network none \
  challenge1a-solution
```

### Challenge 1B Testing
```bash
# Test with collection data
cd Challenge_1b_RF
docker run --rm \
  -v $(pwd)/Collection_1/PDFs:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b-solution
```

## 📈 Future Enhancements

- **Multi-language Support**: Extend to non-English documents
- **Advanced Layout Analysis**: Better handling of complex document layouts
- **Fine-tuned Models**: Domain-specific model training
- **Real-time Processing**: Streaming capabilities for large documents
- **Interactive Interface**: Web-based document analysis tool

---

**Note**: Both solutions are designed to run offline without internet access and comply with all hackathon constraints including execution time, model size, and resource usage limits. 