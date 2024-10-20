import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import string

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------
# 1. Data Loading and Initial Exploration
# ----------------------------------------------------------------------------------

# Load the dataset
data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Fill missing values with empty strings (if any)
data = data.fillna('')

# ----------------------------------------------------------------------------------
# 2. Label Mapping and Encoding
# ----------------------------------------------------------------------------------

# Encode the labels: Convert "Safe Email" to 0 and "Phishing Email" to 1
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
data['Label'] = data['Email Type'].map(label_mapping)

# ----------------------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------------------

# Define functions to extract additional features
def has_url(text):
    return int(bool(re.search(r'http\S+|www\.\S+', text)))

def has_email(text):
    return int(bool(re.search(r'\S+@\S+', text)))

def has_phone(text):
    return int(bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}', text)))

def message_length(text):
    return len(text)

def num_exclamations(text):
    return text.count('!')

def num_questions(text):
    return text.count('?')

def num_uppercase(text):
    return sum(1 for c in text if c.isupper())

def num_digits(text):
    return sum(c.isdigit() for c in text)

# Apply feature extraction
data['has_url'] = data['Email Text'].apply(has_url)
data['has_email'] = data['Email Text'].apply(has_email)
data['has_phone'] = data['Email Text'].apply(has_phone)
data['message_length'] = data['Email Text'].apply(message_length)
data['num_exclamations'] = data['Email Text'].apply(num_exclamations)
data['num_questions'] = data['Email Text'].apply(num_questions)
data['num_uppercase'] = data['Email Text'].apply(num_uppercase)
data['num_digits'] = data['Email Text'].apply(num_digits)

# ----------------------------------------------------------------------------------
# 4. Text Preprocessing Function
# ----------------------------------------------------------------------------------

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Optionally, remove numbers
    # text = re.sub(r'\d+', '', text)
    return text

data['Email Text'] = data['Email Text'].apply(preprocess_text)

# ----------------------------------------------------------------------------------
# 5. Train-Test Split
# ----------------------------------------------------------------------------------

# Define features and target variable
X = data.drop(['Email Type', 'Label'], axis=1)
y = data['Label']

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------------
# 6. Building the Pipeline
# ----------------------------------------------------------------------------------

# Features to be processed
text_features = 'Email Text'
numeric_features = ['has_url', 'has_email', 'has_phone', 'message_length',
                    'num_exclamations', 'num_questions', 'num_uppercase', 'num_digits']

# Text preprocessing and TF-IDF vectorization
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=5,
        ngram_range=(1, 2),
        max_features=10000
    ))
])

# Numeric features scaling
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features),
        ('num', numeric_transformer, numeric_features)
    ]
)

# Create an imbalanced pipeline with SMOTE
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# ----------------------------------------------------------------------------------
# 7. Hyperparameter Tuning with RandomizedSearchCV
# ----------------------------------------------------------------------------------

# Define the parameter grid for Random Forest
param_distributions = {
    'rf__n_estimators': [200, 300, 400],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__min_samples_split': [2, 5, 10]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,                # Limit to 10 iterations for efficiency
    cv=3,                     # 3-fold cross-validation for speed
    scoring='accuracy',       # Optimize for accuracy
    n_jobs=-1,                # Use all available cores
    verbose=2,
    random_state=42
)

# Fit RandomizedSearchCV to the training data
print("\nStarting Randomized Search for Hyperparameter Tuning...")
random_search.fit(X_train, y_train)

# Best parameters
print("\nBest parameters found: ", random_search.best_params_)

# ----------------------------------------------------------------------------------
# 8. Training the Optimized Random Forest Model
# ----------------------------------------------------------------------------------

# Use the best estimator from RandomizedSearchCV
best_pipeline = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_pipeline.predict(X_test)

# ----------------------------------------------------------------------------------
# 9. Model Evaluation
# ----------------------------------------------------------------------------------

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision_non_phishing = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
recall_non_phishing = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_non_phishing = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

precision_phishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_phishing = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_phishing = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\n[Optimized RF] Accuracy: {accuracy:.4f}")
print(f"[Optimized RF] Precision for Non-Phishing (label=0): {precision_non_phishing:.4f}")
print(f"[Optimized RF] Recall for Non-Phishing (label=0): {recall_non_phishing:.4f}")
print(f"[Optimized RF] F1-score for Non-Phishing (label=0): {f1_non_phishing:.4f}")

print(f"[Optimized RF] Precision for Phishing (label=1): {precision_phishing:.4f}")
print(f"[Optimized RF] Recall for Phishing (label=1): {recall_phishing:.4f}")
print(f"[Optimized RF] F1-score for Phishing (label=1): {f1_phishing:.4f}")

# Classification Report
print("\n[Optimized RF] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Phishing', 'Phishing']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n[Optimized RF] Confusion Matrix:")
print(conf_matrix)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Phishing', 'Phishing'],
            yticklabels=['Non-Phishing', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized Random Forest - Confusion Matrix')
plt.show()

# ----------------------------------------------------------------------------------
# 10. Cross-Validation for Robust Evaluation
# ----------------------------------------------------------------------------------

# Define Stratified K-Fold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Evaluate the optimized Random Forest using cross-validation
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
print(f"\n[Cross-Validation] Random Forest Accuracy Scores: {cv_scores}")
print(f"[Cross-Validation] Mean Accuracy Score: {cv_scores.mean():.4f}")
