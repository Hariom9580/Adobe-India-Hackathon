import os
import json
import datetime
from extract_sections import extract_all_sections
from feature_engineering import compute_features
from model import predict_relevance

INPUT_PDF_DIR = "/app/input"
OUTPUT_JSON = "/app/output/challenge1b_output.json"

if __name__ == "__main__":
    if not os.path.exists(INPUT_PDF_DIR):
        print(f"Input directory {INPUT_PDF_DIR} not found")
        exit(1)
    
    os.makedirs("/app/output", exist_ok=True)
    
    persona = os.environ.get("PERSONA", "Researcher")
    job = os.environ.get("JOB", "Analyze documents for key insights")
    
    print(f"Processing PDFs from {INPUT_PDF_DIR}")
    print(f"Persona: {persona}")
    print(f"Job: {job}")
    
    pdf_sections = extract_all_sections(INPUT_PDF_DIR)
    all_sections = []
    for doc, sections in pdf_sections.items():
        for s in sections:
            s['document'] = doc
            all_sections.append(s)
    
    if not all_sections:
        print("No sections extracted from PDFs")
        exit(1)
    
    features_df = compute_features(all_sections, persona, job)
    scores = predict_relevance(features_df)
    features_df['relevance_score'] = scores
    features_df = features_df.sort_values('relevance_score', ascending=False)
    
    metadata = {
        "input_documents": list(pdf_sections.keys()),
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.datetime.now().isoformat()
    }
    
    extracted_sections = []
    for i, row in features_df.head(5).iterrows():
        extracted_sections.append({
            "document": row['document'],
            "section_title": row['section_title'],
            "importance_rank": i+1,
            "page_number": int(row['page_number'])
        })
    
    subsection_analysis = []
    for i, row in features_df.head(5).iterrows():
        subsection_analysis.append({
            "document": row['document'],
            "refined_text": row['text'][:800],
            "page_number": int(row['page_number'])
        })
    
    output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"Output written to {OUTPUT_JSON}")
    print(f"Processed {len(all_sections)} sections from {len(pdf_sections)} documents") 