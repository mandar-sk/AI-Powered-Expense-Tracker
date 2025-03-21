
import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# ======================
# Configuration
# ======================
MODEL_VERSION = "1.0.0"
DATA_PATH = "data/train_data.csv"
MODEL_PATH = f"models/transaction_classifier_v{MODEL_VERSION}.joblib"

# ======================
# Text Processing
# ======================
def custom_tokenizer(text: str) -> list:
    """More robust tokenizer with numeric handling"""
    text = str(text).lower()
    # Preserve transaction codes and amounts
    tokens = re.findall(r'\b[a-z0-9@/-]+(?:\.[0-9]+)?\b', text)
    return tokens or ['missing_tx']  # Fallback token

def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add text validation"""
    df['processed_text'] = df['Narration'].apply(
        lambda x: ' '.join(custom_tokenizer(x)) or 'missing_tx'
    )
    
    # Remove empty texts
    df = df[df['processed_text'].str.strip() != '']
    
    return df

# ======================
# Model Training
# ======================
def train_and_save_model():
    # Create models directory
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    df = preprocess_text(df)
    
    # Clean categories
    df['category'] = df['category'].str.strip().replace(
        {'Accomadation': 'Accommodation'}
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    
    # Build model pipeline
    """model = make_pipeline(
        TfidfVectorizer(
            tokenizer=custom_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2)),
        LogisticRegression(
            max_iter=20000,
            solver='lbfgs',  
            C=0.5,           
            class_weight='balanced',
            random_state=42)
        )"""
    
    model = make_pipeline(
        TfidfVectorizer(
            tokenizer=custom_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=3000,
            min_df=2
        ),
        LinearSVC(
            class_weight='balanced',
            max_iter=10000,
            dual=False,        # Recommended when n_samples < n_features
            tol=1e-3,          # Looser tolerance
            random_state=42,
            verbose=1         # Show training progress
        )
    )
    # Train model
    model.fit(df['processed_text'], df['category_encoded'])
    
    # Save model with metadata
    joblib.dump(
        {
            'model': model,
            'label_encoder': label_encoder,
            'version': MODEL_VERSION
        },
        MODEL_PATH
    )
    print(f"✅ Model saved to {MODEL_PATH}")

# ======================
# Prediction Interface
# ======================
class TransactionClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        
    def load(self):
        """Load trained model"""
        artifacts = joblib.load(MODEL_PATH)
        self.model = artifacts['model']
        self.label_encoder = artifacts['label_encoder']
    
    def predict(self, description: str) -> str:
        """Predict category for a transaction"""
        processed = ' '.join(custom_tokenizer(description))
        prediction = self.model.predict([processed])
        return self.label_encoder.inverse_transform(prediction)[0]

if __name__ == "__main__":
    # Train and save model when run directly
    train_and_save_model()