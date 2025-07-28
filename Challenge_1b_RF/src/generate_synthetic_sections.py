import pandas as pd
import random
import numpy as np

section_titles = [
    "Introduction", "Background", "Methodology", "Results", "Discussion", "Conclusion", "Summary", "Analysis", "Overview", "Key Points",
    "Recommendations", "Appendix", "References", "Abstract", "Case Study", "Findings", "Future Work", "Limitations", "Objectives", "Scope"
]
documents = [f"Document_{i+1}.pdf" for i in range(10)]
persona = "Sample Persona"
job = "Sample Job Description"

def random_section():
    title = random.choice(section_titles)
    text = f"This is a sample text for the section {title}. It contains information relevant to the topic."
    page_number = random.randint(1, 20)
    tfidf_sim_persona = round(random.uniform(0, 1), 3)
    tfidf_sim_job = round(random.uniform(0, 1), 3)
    section_length = random.randint(50, 500)
    title_length = len(title)
    relevance = random.randint(0, 1)
    return {
        "document": random.choice(documents),
        "section_title": title,
        "text": text,
        "page_number": page_number,
        "tfidf_sim_persona": tfidf_sim_persona,
        "tfidf_sim_job": tfidf_sim_job,
        "section_length": section_length,
        "title_length": title_length,
        "relevance": relevance
    }

def generate_synthetic_sections(n=500):
    rows = [random_section() for _ in range(n)]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_synthetic_sections(500)
    df.to_csv("synthetic_sections.csv", index=False)
    print("Generated synthetic_sections.csv with 500 rows") 