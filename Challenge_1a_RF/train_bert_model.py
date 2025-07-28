import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pathlib import Path
import json
import fitz
from typing import List, Dict, Any

class HeadingDataset(Dataset):
    """Dataset for training BERT model on heading classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_training_data():
    """Create synthetic training data for heading classification."""
    training_data = []
    
    # Sample headings and body text
    headings = [
        "Introduction to Machine Learning",
        "What is Artificial Intelligence",
        "Deep Learning Fundamentals",
        "Neural Networks Explained",
        "Data Preprocessing Techniques",
        "Model Evaluation Methods",
        "Feature Engineering",
        "Hyperparameter Tuning",
        "Cross-Validation Strategies",
        "Ensemble Methods",
        "Natural Language Processing",
        "Computer Vision Applications",
        "Reinforcement Learning",
        "Transfer Learning",
        "Model Deployment",
        "Ethics in AI",
        "Future of Machine Learning",
        "Industry Applications",
        "Research Directions",
        "Conclusion and Next Steps"
    ]
    
    body_texts = [
        "This chapter provides an overview of the fundamental concepts.",
        "The following section discusses various approaches and methodologies.",
        "We will explore different techniques and their applications.",
        "This analysis shows the results of our experiments.",
        "The data indicates several important trends and patterns.",
        "Our findings suggest several implications for future research.",
        "The methodology section describes our experimental setup.",
        "Results are presented in the following tables and figures.",
        "Discussion of the implications follows in the next section.",
        "Conclusions are drawn based on the evidence presented."
    ]
    
    # Create training examples
    for heading in headings:
        # Add heading as H1, H2, H3
        training_data.append({
            'text': heading,
            'label': 'H1',
            'font_size': 18,
            'is_bold': True
        })
        
        training_data.append({
            'text': heading,
            'label': 'H2', 
            'font_size': 16,
            'is_bold': True
        })
        
        training_data.append({
            'text': heading,
            'label': 'H3',
            'font_size': 14,
            'is_bold': False
        })
    
    # Add title examples
    titles = [
        "Machine Learning Fundamentals",
        "Artificial Intelligence: A Comprehensive Guide",
        "Deep Learning for Beginners",
        "Data Science Handbook",
        "AI and the Future of Technology"
    ]
    
    for title in titles:
        training_data.append({
            'text': title,
            'label': 'Title',
            'font_size': 20,
            'is_bold': True
        })
    
    # Add body text examples
    for body in body_texts:
        training_data.append({
            'text': body,
            'label': 'Body',
            'font_size': 12,
            'is_bold': False
        })
    
    return training_data

def train_bert_model():
    """Train BERT model for heading classification."""
    print("Creating training data...")
    training_data = create_training_data()
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Label encoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])
    
    # Create dataset
    dataset = HeadingDataset(df['text'].tolist(), labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True
    )
    
    # Training setup
    device = torch.device('cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(3):  # Small number of epochs for quick training
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.4f}")
    
    # Save model and tokenizer
    print("Saving model...")
    model_path = Path("bert_heading_model")
    model_path.mkdir(exist_ok=True)
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save label encoder
    import joblib
    joblib.dump(label_encoder, model_path / "label_encoder.joblib")
    
    print("Training completed! Model saved to 'bert_heading_model' directory.")
    
    return model, tokenizer, label_encoder

if __name__ == "__main__":
    train_bert_model() 