import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_features(sections, persona, job):
    """
    Given a list of section dicts, persona string, and job string,
    returns a DataFrame with features for each section.
    Features: tfidf_sim_persona, tfidf_sim_job, section_length, page_number, title_length
    """
    df = pd.DataFrame(sections)
    # TF-IDF similarity
    tfidf = TfidfVectorizer().fit([persona, job] + list(df['section_title']))
    persona_vec = tfidf.transform([persona]).toarray()[0]
    job_vec = tfidf.transform([job]).toarray()[0]
    section_vecs = tfidf.transform(df['section_title']).toarray()
    # Cosine similarity
    def cosine_sim(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    import numpy as np
    df['tfidf_sim_persona'] = [cosine_sim(vec, persona_vec) for vec in section_vecs]
    df['tfidf_sim_job'] = [cosine_sim(vec, job_vec) for vec in section_vecs]
    df['section_length'] = df['text'].apply(len)
    df['title_length'] = df['section_title'].apply(len)
    # Normalize page number
    df['page_number'] = df['page_number']
    return df[['document','section_title','text','page_number','tfidf_sim_persona','tfidf_sim_job','section_length','title_length']]

if __name__ == "__main__":
    # Example usage
    import sys, json
    if len(sys.argv) < 4:
        print("Usage: python feature_engineering.py <sections.json> <persona> <job>")
    else:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            sections = json.load(f)
        persona = sys.argv[2]
        job = sys.argv[3]
        features = compute_features(sections, persona, job)
        print(features.head()) 