import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

def train_random_forest(X, y, model_path):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    return clf

def load_model(model_path):
    return joblib.load(model_path)

def predict_relevance(features_df, model_path=None):
    # If no model, use a simple rule: relevant if tfidf_sim_persona or tfidf_sim_job > 0.2
    if model_path and os.path.exists(model_path):
        clf = load_model(model_path)
        X = features_df[['tfidf_sim_persona','tfidf_sim_job','section_length','title_length','page_number']]
        scores = clf.predict_proba(X)[:,1] if hasattr(clf, 'predict_proba') else clf.predict(X)
        return scores
    else:
        # Simple rule-based fallback
        return (features_df['tfidf_sim_persona'] + features_df['tfidf_sim_job']) / 2

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model.py <features.csv> [model_path]")
    else:
        features_df = pd.read_csv(sys.argv[1])
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        scores = predict_relevance(features_df, model_path)
        print(scores) 