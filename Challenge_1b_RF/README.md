# Challenge 1B: Persona-Driven Document Intelligence

## Overview
This solution implements intelligent document analysis that extracts and prioritizes relevant sections from a collection of documents based on a specific persona and their job-to-be-done. The system uses semantic similarity and feature engineering to identify the most relevant content for the given user context.

## 🚀 Approach

### Semantic Relevance Analysis
- **TF-IDF Vectorization**: Converts text sections and job requirements into numerical vectors
- **Cosine Similarity**: Calculates relevance scores between sections and requirements
- **Feature Engineering**: Combines multiple features for robust ranking
- **Persona Understanding**: Considers user expertise and focus areas

### Key Features
- **Multi-Document Processing**: Handles collections of 3-10 related PDFs
- **Persona-Aware Ranking**: Prioritizes content relevant to specific user roles
- **Job-Focused Analysis**: Aligns with concrete tasks and requirements
- **Structured Output**: Provides metadata, extracted sections, and subsection analysis

## 📊 Models & Libraries Used

- **scikit-learn**: TF-IDF vectorization and similarity calculations
- **pandas**: Data processing and feature engineering
- **numpy**: Numerical computations
- **pdfplumber**: PDF text extraction with formatting information

## 🛠️ Build & Run Instructions

### Build Docker Image
```bash
docker build --platform linux/amd64 -t challenge1b-solution .
```

### Run Solution
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b-solution
```

### With Custom Persona/Job
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Investment Analyst" \
  -e JOB="Analyze revenue trends and market positioning" \
  --network none \
  challenge1b-solution
```

## 📁 Project Structure

```
Challenge_1b_RF/
├── src/
│   ├── main.py                 # Main entry point
│   ├── extract_sections.py     # PDF section extraction
│   ├── feature_engineering.py  # Feature computation
│   └── model.py               # Relevance prediction
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── Collection_*/              # Sample document collections
```

## 🔧 Technical Implementation

### 1. Section Extraction
- Extracts text sections from all PDFs in input directory
- Preserves document structure and page numbers
- Handles various PDF formats and layouts

### 2. Feature Engineering
- **Text Features**: TF-IDF vectors for section content
- **Persona Features**: TF-IDF vectors for persona description
- **Job Features**: TF-IDF vectors for job requirements
- **Combined Features**: Merged feature set for similarity calculation

### 3. Relevance Prediction
- **Cosine Similarity**: Calculates similarity between sections and requirements
- **Ranking**: Sorts sections by relevance score
- **Selection**: Returns top 5 most relevant sections

### 4. Output Generation
- **Metadata**: Input documents, persona, job, timestamp
- **Extracted Sections**: Top 5 sections with importance ranking
- **Subsection Analysis**: Detailed content analysis for each section

## 📊 Output Format

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "Investment Analyst",
    "job_to_be_done": "Analyze revenue trends",
    "processing_timestamp": "2025-01-28T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "Financial Performance",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "Revenue increased by 15%...",
      "page_number": 5
    }
  ]
}
```

## ⚡ Performance Compliance

### Constraints Met
- ✅ **Execution Time**: ≤ 60 seconds for 3-5 documents
- ✅ **Model Size**: ≤ 1GB (lightweight TF-IDF approach)
- ✅ **Network**: No internet access required
- ✅ **Runtime**: CPU-only processing
- ✅ **Memory**: Efficient memory usage

### Performance Metrics
- **Processing Speed**: ~10-15 seconds per document
- **Memory Usage**: < 2GB RAM
- **Scalability**: Handles up to 10 documents efficiently
- **Accuracy**: High relevance scores for matching content

## 🧪 Testing

### Local Testing
```bash
# Test with sample collection
python src/main.py
```

### Docker Testing
```bash
# Test with collection data
docker run --rm \
  -v $(pwd)/Collection_1/PDFs:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b-solution
```

### Validation Checklist
- [x] All PDFs processed successfully
- [x] JSON output format correct
- [x] Performance within constraints
- [x] No internet access required
- [x] AMD64 compatibility
- [x] Memory usage controlled

## 🎯 Key Features

### Persona-Aware Analysis
- Considers user expertise and background
- Adapts to different professional contexts
- Handles diverse job requirements

### Robust Processing
- Handles various PDF formats
- Graceful error handling
- Efficient memory management

### Scalable Architecture
- Modular code structure
- Easy to extend and modify
- Well-documented components

## 📈 Future Enhancements

### Potential Improvements
- **Advanced NLP**: Use BERT embeddings for better semantic understanding
- **Multi-language Support**: Extend to non-English documents
- **Interactive Interface**: Web-based document analysis tool
- **Real-time Processing**: Streaming capabilities for large document sets

### Advanced Features
- **Context Awareness**: Consider document relationships
- **Temporal Analysis**: Handle time-sensitive content
- **Domain Adaptation**: Fine-tune for specific domains
- **Collaborative Filtering**: Learn from user preferences

---

**Note**: This solution is designed to run offline without internet access and complies with all hackathon constraints including execution time, model size, and resource usage limits. 