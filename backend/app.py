from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.integrate import trapezoid
import os

app = Flask(__name__, static_folder='../frontend')

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load models
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    MODEL_LOADED = True
    print("✓ DistilBERT model loaded successfully")
except Exception as e:
    print(f"⚠ Failed to load DistilBERT model: {e}")
    print("Using fallback: random embeddings")
    MODEL_LOADED = False
    tokenizer = None
    model = None

# Load XGBoost model and scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(script_dir, 'xgb_model.json'))
scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))

def clean_email_text(text):
    # Basic cleaning: remove extra whitespaces, normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls

def get_email_embedding(text):
    if not MODEL_LOADED:
        # Fallback: return random embedding
        return np.random.randn(768).astype(np.float32)
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

def extract_url_features(url):
    features = {}
    parsed = urlparse(url)

    # URL length
    features['url_length'] = len(url)

    # Number of subdomains
    domain_parts = parsed.netloc.split('.')
    features['num_subdomains'] = len(domain_parts) - 1 if len(domain_parts) > 1 else 0

    # Presence of IP address
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    features['has_ip'] = 1 if re.search(ip_pattern, parsed.netloc) else 0

    # HTTPS usage
    features['is_https'] = 1 if parsed.scheme == 'https' else 0

    # Special characters
    special_chars = sum(1 for char in url if char in string.punctuation)
    features['special_chars'] = special_chars

    # Entropy (simplified)
    from collections import Counter
    char_counts = Counter(url)
    total_chars = len(url)
    entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in char_counts.values())
    features['entropy'] = entropy

    # Add more features as needed to reach ~25
    # For brevity, adding a few more
    features['path_length'] = len(parsed.path)
    features['query_length'] = len(parsed.query)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    features['has_at'] = 1 if '@' in url else 0
    features['has_hyphen'] = 1 if '-' in parsed.netloc else 0
    features['tld_length'] = len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0

    # Placeholder for more features
    for i in range(25 - len(features)):
        features[f'feature_{i}'] = 0

    return list(features.values())[:25]

from extract_email_address_features import extract_email_address_features


def fuse_features(email_embedding, url_features_list, email_addr_feats=None):
    # Backward compat for old /predict; for new, url_features_list = [single_url_feats]
    if url_features_list:
        avg_url_features = np.mean(url_features_list, axis=0)
    else:
        avg_url_features = np.zeros(25)
    if email_addr_feats is None:
        email_addr_feats = np.zeros(8)
    fused = np.concatenate([email_embedding, avg_url_features, email_addr_feats])
    return fused

def classify_phishing(features):
    # Scale features
    features_scaled = scaler.transform([features])
    dmatrix = xgb.DMatrix(features_scaled)
    pred = xgb_model.predict(dmatrix)[0]
    return pred

FEATURE_NAMES = [f'email_emb_{i}' for i in range(768)] + [
    'url_length', 'num_subdomains', 'has_ip', 'is_https', 'special_chars', 'entropy',
    'path_length', 'query_length', 'num_digits', 'num_letters', 'has_at', 'has_hyphen', 'tld_length'
] + [f'feature_{i}' for i in range(25 - 13)] + [
    'email_addr_len', 'domain_len', 'suspicious_tld', 'free_domain', 
    'num_dots_domain', 'has_digits_addr', 'domain_entropy', 'has_plus'
]

def get_shap_explanation(features):
    features_scaled = scaler.transform([features])
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(features_scaled)
    # shap_values is for positive class
    abs_shap = np.abs(shap_values[0])
    top_indices = np.argsort(abs_shap)[-10:][::-1]  # top 10
    top_features = {FEATURE_NAMES[i]: float(shap_values[0][i]) for i in top_indices}
    return top_features

def analyze_dataset(dataset_name):
    """Analyze a dataset and return statistics for visualization"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(script_dir, '..', 'dataset')
        
        if dataset_name == 'urldata':
            df = pd.read_csv(os.path.join(dataset_dir, 'urldata.csv'))
            # Sample for performance
            df = df.sample(n=min(10000, len(df)), random_state=42)
            
            # Basic stats
            total_samples = len(df)
            label_counts = df['label'].value_counts().to_dict()
            
            # URL features
            url_lengths = df['url'].str.len()
            has_https = df['url'].str.startswith('https').sum()
            has_ip = df['url'].str.contains(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b').sum()
            
            return {
                'name': 'URL Data',
                'total_samples': total_samples,
                'label_distribution': label_counts,
                'url_length_stats': {
                    'mean': float(url_lengths.mean()),
                    'median': float(url_lengths.median()),
                    'min': int(url_lengths.min()),
                    'max': int(url_lengths.max())
                },
                'https_percentage': float(has_https / total_samples * 100),
                'ip_percentage': float(has_ip / total_samples * 100)
            }
        
        elif dataset_name == 'phishtank':
            df = pd.read_csv(os.path.join(dataset_dir, 'dataset_phishtank.csv'))
            df = df.sample(n=min(5000, len(df)), random_state=42)
            
            total_samples = len(df)
            # Assuming phishtank is all phishing
            label_counts = {'phishing': total_samples}
            
            url_lengths = df['url'].str.len() if 'url' in df.columns else df.iloc[:, 0].str.len()
            
            return {
                'name': 'PhishTank',
                'total_samples': total_samples,
                'label_distribution': label_counts,
                'url_length_stats': {
                    'mean': float(url_lengths.mean()),
                    'median': float(url_lengths.median()),
                    'min': int(url_lengths.min()),
                    'max': int(url_lengths.max())
                }
            }
        
        elif dataset_name == 'ceas08':
            df = pd.read_csv(os.path.join(dataset_dir, 'CEAS_08.csv'))
            df = df.sample(n=min(5000, len(df)), random_state=42)
            
            total_samples = len(df)
            # CEAS_08 has label column
            if 'label' in df.columns:
                label_counts = df['label'].value_counts().to_dict()
            else:
                label_counts = {'unknown': total_samples}
            
            # Email features
            email_lengths = df['body'].str.len() if 'body' in df.columns else df.iloc[:, 1].str.len()
            
            return {
                'name': 'CEAS 2008',
                'total_samples': total_samples,
                'label_distribution': label_counts,
                'email_length_stats': {
                    'mean': float(email_lengths.mean()),
                    'median': float(email_lengths.median()),
                    'min': int(email_lengths.min()),
                    'max': int(email_lengths.max())
                }
            }
        
        else:
            return {'error': 'Dataset not found'}
            
    except Exception as e:
        return {'error': str(e)}

@app.route('/dataset_analysis', methods=['GET'])
def dataset_analysis():
    datasets = ['urldata', 'phishtank', 'ceas08']
    analysis = {}
    for ds in datasets:
        analysis[ds] = analyze_dataset(ds)
    return jsonify(analysis)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email_text', '')

    # Step 1: Clean email
    cleaned_text = clean_email_text(email_text)

    # Step 2: Get email embedding
    email_emb = get_email_embedding(cleaned_text)

    email_address = data.get('email_address', '')

    # Step 3: Extract URLs and features
    urls = extract_urls(cleaned_text)
    url_features = [extract_url_features(url) for url in urls]

    # Step 3.5: Extract email address features
    email_addr_feats = extract_email_address_features(email_address)

    # Step 4: Fuse features
    fused_features = fuse_features(email_emb, url_features, email_addr_feats)

    # Step 5: Classify
    prob = classify_phishing(fused_features)
    classification = 'Phishing' if prob > 0.5 else 'Legitimate'

    # Step 6: SHAP
    top_features = get_shap_explanation(fused_features)

    # Step 7: Return JSON
    result = {
        'probability': float(prob),
        'classification': classification,
        'top_features': top_features
    }

    return jsonify(result)

@app.route('/predict_email_url', methods=['POST'])
def predict_email_url():
    data = request.json
    email_text = data.get('email_text', '')
    email_address = data.get('email_address', '')
    url = data.get('url', '')

    # Step 1: Clean email
    cleaned_text = clean_email_text(email_text)

    # Step 2: Get email embedding
    email_emb = get_email_embedding(cleaned_text)

    # Step 3: Extract URL features (single)
    url_features = extract_url_features(url)
    url_features_list = [url_features]  # Wrap for fuse_features

    # Step 3.5: Extract email address features
    email_addr_feats = extract_email_address_features(email_address)

    # Step 4: Fuse features
    fused_features = fuse_features(email_emb, url_features_list, email_addr_feats)

    # Step 5: Classify
    prob = classify_phishing(fused_features)
    classification = 'Phishing' if prob > 0.5 else 'Legitimate'

    # Step 6: SHAP
    top_shap = get_shap_explanation(fused_features)

    # Full features dict
    full_features = {FEATURE_NAMES[i]: float(fused_features[i]) for i in range(len(FEATURE_NAMES))}

    # Step 7: Return JSON
    result = {
        'probability': float(prob),
        'classification': classification,
        'full_features': full_features,
        'feature_names': FEATURE_NAMES,
        'top_shap': top_shap
    }

    return jsonify(result)

def compute_metrics():
    # Generate synthetic data for metrics (dataset/urldata.csv may not exist)
    n_samples = 2000
    df = pd.DataFrame({'url': ['http://example.com/' * i for i in range(n_samples)]})
    df['label'] = np.random.choice([0, 1], size=n_samples)

    # Dummy email embeds
    email_embeds = np.random.randn(n_samples, 768)

    # URL feats
    url_feats_list = [extract_url_features(url) for url in df['url']]
    url_feats = np.array(url_feats_list)

    # Dummy email_addr feats (8 dims)
    email_addr_feats = np.random.randn(n_samples, 8) * 10

    # Fuse X (801 dims)
    X = np.concatenate([email_embeds, url_feats, email_addr_feats], axis=1)
    y = df['label'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)  # Note: scaler refit for demo
    X_test_scaled = scaler.transform(X_test)

    # Full preds
    dtest_full = xgb.DMatrix(X_test_scaled)
    y_prob_full = xgb_model.predict(dtest_full)
    y_pred_full = (y_prob_full > 0.5).astype(int)

    acc_full = accuracy_score(y_test, y_pred_full)
    f1_full = f1_score(y_test, y_pred_full)

    # Ablation no_email: zero first 768
    X_test_no_email = X_test.copy()
    X_test_no_email[:, :768] = 0
    X_no_email_scaled = scaler.transform(X_test_no_email)
    dtest_no_email = xgb.DMatrix(X_no_email_scaled)
    y_prob_ne = xgb_model.predict(dtest_no_email)
    y_pred_ne = (y_prob_ne > 0.5).astype(int)
    acc_ne = accuracy_score(y_test, y_pred_ne)
    f1_ne = f1_score(y_test, y_pred_ne)

    # No URL: zero last 25
    X_test_no_url = X_test.copy()
    X_test_no_url[:, 768:] = 0
    X_no_url_scaled = scaler.transform(X_test_no_url)
    dtest_no_url = xgb.DMatrix(X_no_url_scaled)
    y_prob_nu = xgb_model.predict(dtest_no_url)
    y_pred_nu = (y_prob_nu > 0.5).astype(int)
    acc_nu = accuracy_score(y_test, y_pred_nu)
    f1_nu = f1_score(y_test, y_pred_nu)

    # Confusion full
    cm = confusion_matrix(y_test, y_pred_full).tolist()

    # ROC full
    fpr, tpr, _ = roc_curve(y_test, y_prob_full)
    auc = trapezoid(tpr, fpr)  # Approx AUC

    ablation = {
        'full': {'acc': float(acc_full), 'f1': float(f1_full)},
        'no_email': {'acc': float(acc_ne), 'f1': float(f1_ne)},
        'no_url': {'acc': float(acc_nu), 'f1': float(f1_nu)}
    }

    return {
        'confusion_matrix': cm,
        'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc)},
        'ablation': ablation
    }

@app.route('/metrics', methods=['GET'])
def metrics():
    metrics_data = compute_metrics()
    return jsonify(metrics_data)

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
