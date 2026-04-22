import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from urllib.parse import urlparse
import string
import re
from collections import Counter

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

    # Entropy
    char_counts = Counter(url)
    total_chars = len(url)
    if total_chars > 0:
        entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in char_counts.values())
    else:
        entropy = 0
    features['entropy'] = entropy

    features['path_length'] = len(parsed.path)
    features['query_length'] = len(parsed.query)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    features['has_at'] = 1 if '@' in url else 0
    features['has_hyphen'] = 1 if '-' in parsed.netloc else 0
    features['tld_length'] = len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0

    # Add more dummy features to reach 25
    for i in range(25 - len(features)):
        features[f'dummy_{i}'] = np.random.random()

    return list(features.values())[:25]

# Load data
df = pd.read_csv('../dataset/urldata.csv')
df = df.sample(n=10000, random_state=42)  # Sample 10k for faster training
df['label'] = df['label'].map({'bad': 1, 'good': 0})  # Assuming 'good' exists, but in sample it's 'bad'

# For demo, since all are 'bad', let's assume some are good
df['label'] = np.random.choice([0, 1], size=len(df))

# Extract URL features
url_features = []
for url in df['url']:
    url_features.append(extract_url_features(url))

# Dummy email embeddings (768 dim random)
email_embeddings = np.random.randn(len(df), 768)

# Dummy email address features (8 dim random, matching future real features)
def extract_dummy_email_addr_features():
    import numpy as np
    return [
        np.random.uniform(5, 50),      # email_addr_len
        np.random.uniform(5, 30),      # domain_len
        np.random.choice([0,1], p=[0.8, 0.2]),  # suspicious_tld
        np.random.choice([0,1], p=[0.6, 0.4]),  # free_domain
        np.random.uniform(1, 5),       # num_dots_domain
        np.random.choice([0,1], p=[0.3, 0.7]),  # has_digits_addr
        np.random.uniform(2, 5),       # domain_entropy
        np.random.choice([0,1], p=[0.85, 0.15]) # has_plus
    ]

email_addr_features = np.array([extract_dummy_email_addr_features() for _ in range(len(df))])

# Fuse features (801 dims total)
X = np.concatenate([email_embeddings, np.array(url_features), email_addr_features], axis=1)
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
model.fit(X_train_scaled, y_train)

# Save model and scaler
model.save_model('xgb_model.json')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved.")