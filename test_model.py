# test_model.py
import os
import re
import time
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB

# ======================
# Configuration
# ======================
MODEL_DIR = "models"
DATA_PATH = "data/train_data.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ======================
# Text Processing
# ======================
def custom_tokenizer(text: str) -> list:
    """Tokenize transaction descriptions into meaningful units"""
    text = str(text).lower()
    return re.findall(r'\b[a-z0-9@/-]+\b', text) or ['missing_tx']

def preprocess_transaction_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare transaction text data"""
    # Handle missing values
    df['Narration'] = df['Narration'].fillna('missing_description')
    
    # Create processed text with transaction type markers
    transaction_types = ['UPI', 'NEFT', 'IMPS', 'ATM', 'POS', 'FD', 'CR', 'DR']
    
    df['processed_text'] = df['Narration'].apply(
        lambda x: ' '.join(custom_tokenizer(x))
    )
    
    # Add transaction type indicators
    for t_type in transaction_types:
        pattern = re.compile(rf'\b{t_type}\b', re.IGNORECASE)
        markers = df['Narration'].apply(
            lambda x: f'TXN_TYPE_{t_type}' if pattern.search(str(x)) else ''
        )
        df['processed_text'] += ' ' + markers
    
    # Clean whitespace
    df['processed_text'] = (
        df['processed_text']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    
    return df

# ======================
# Model Evaluation
# ======================
def load_and_preprocess_data():
    """Load and preprocess the training data"""
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess text
    df = preprocess_transaction_text(df)
    
    # Clean categories
    df['category'] = df['category'].str.strip().replace(
        {'Accomadation': 'Accommodation'}
    )
    
    # Filter empty texts
    df = df[df['processed_text'].str.strip() != '']
    
    return df

def evaluate_model(model, X_test, y_test):
    """Evaluate model on multiple metrics"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'inference_time': inference_time,
        'classification_report': classification_report(y_test, y_pred)
    }

def compare_models():
    """Main function to compare model performance"""
    # Load and prepare data
    df = load_and_preprocess_data()
    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    X = df['processed_text']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Define models to compare
    models = {
        'Logistic Regression': make_pipeline(
            TfidfVectorizer(
                tokenizer=custom_tokenizer,
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2
            ),
            LogisticRegression(
                max_iter=10000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=RANDOM_STATE
            )
        ),
        'Linear SVM': make_pipeline(
            TfidfVectorizer(
                tokenizer=custom_tokenizer,
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2
            ),
            LinearSVC(
                class_weight='balanced',
                dual=False,
                max_iter=10000,
                random_state=RANDOM_STATE
            )
        ),
        'XGBoost': make_pipeline(
            TfidfVectorizer(
                tokenizer=custom_tokenizer,
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2
            ),
            XGBClassifier(
                objective='multi:softmax',
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=RANDOM_STATE
            )
        ),
        'Random Forest': make_pipeline(
            TfidfVectorizer(
                tokenizer=custom_tokenizer,
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2
            ),
            RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        ),
        'Naive Bayes': make_pipeline(
            TfidfVectorizer(
                tokenizer=custom_tokenizer,
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2
            ),
            MultinomialNB()
        )
    }
    
    # Evaluation results storage
    results = []
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"\n⚡ Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model_name'] = name
        metrics['train_time'] = train_time
        results.append(metrics)
        
        print(f"✅ {name} evaluation complete")
        print(f"⏱️ Train time: {train_time:.2f}s")
        print(f"🏆 Best metric (F1-weighted): {metrics['f1_weighted']:.4f}\n")
    
    # Create comparison report
    report_df = pd.DataFrame(results).sort_values('f1_weighted', ascending=False)
    report_df = report_df[[
        'model_name',
        'f1_weighted',
        'accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'train_time',
        'inference_time'
    ]]
    
    # Save best model
    best_model_name = report_df.iloc[0]['model_name']
    best_model = models[best_model_name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            'pipeline': best_model,
            'label_encoder': le,
            'metrics': report_df.to_dict()
        },
        os.path.join(MODEL_DIR, f"best_model.joblib")
    )
    
    return report_df

if __name__ == "__main__":
    comparison_report = compare_models()
    print("\n🔍 Model Comparison Report:")
    print(comparison_report.to_markdown(index=False))
    
    # Save full report
    comparison_report.to_csv("model_comparison_report.csv", index=False)
    print("\n📄 Full report saved to model_comparison_report.csv")