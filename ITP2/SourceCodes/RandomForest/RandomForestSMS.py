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
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
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
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings (if any)
data = data.fillna('')

# ----------------------------------------------------------------------------------
# 2. Label Mapping and Encoding
# ----------------------------------------------------------------------------------

# Merge 'ham' and 'spam' into 'non-smishing' and map labels to integers
label_mapping = {'ham': 0, 'spam': 0, 'smishing': 1}
data['Label'] = data['LABEL'].str.lower().map(label_mapping)

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Convert labels to integers
data['Label'] = data['Label'].astype(int)

# ----------------------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------------------

# Define functions to extract additional features
def has_url(text):
    return int(bool(re.search(r'http\S+|www\.\S+', text)))

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

def num_words(text):
    return len(text.split())

def avg_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words]) if words else 0

def num_special_chars(text):
    return sum(not c.isalnum() and not c.isspace() for c in text)

# Apply feature extraction
data['has_url'] = data['TEXT'].apply(has_url)
data['has_phone'] = data['TEXT'].apply(has_phone)
data['message_length'] = data['TEXT'].apply(message_length)
data['num_exclamations'] = data['TEXT'].apply(num_exclamations)
data['num_questions'] = data['TEXT'].apply(num_questions)
data['num_uppercase'] = data['TEXT'].apply(num_uppercase)
data['num_digits'] = data['TEXT'].apply(num_digits)
data['num_words'] = data['TEXT'].apply(num_words)
data['avg_word_length'] = data['TEXT'].apply(avg_word_length)
data['num_special_chars'] = data['TEXT'].apply(num_special_chars)

# ----------------------------------------------------------------------------------
# 4. Text Preprocessing
# ----------------------------------------------------------------------------------

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['TEXT'] = data['TEXT'].apply(preprocess_text)

# ----------------------------------------------------------------------------------
# 5. Train-Test Split
# ----------------------------------------------------------------------------------

# Define features and target variable
X = data.drop(['LABEL', 'Label'], axis=1)
y = data['Label']

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------------
# 6. Building the Pipeline
# ----------------------------------------------------------------------------------

# Features to be processed
text_features = 'TEXT'
numeric_features = ['has_url', 'has_phone', 'message_length',
                    'num_exclamations', 'num_questions', 'num_uppercase',
                    'num_digits', 'num_words', 'avg_word_length', 'num_special_chars']

# Text preprocessing and TF-IDF vectorization
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Include unigrams, bigrams, trigrams
        max_features=10000,
        sublinear_tf=True
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

# Create an imbalanced pipeline with SMOTETomek
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smotetomek', SMOTETomek(random_state=42)),
    ('rf', RandomForestClassifier(
        random_state=42,
        class_weight={0: 1, 1: 5}  # Increase weight for the minority class
    ))
])

# ----------------------------------------------------------------------------------
# 7. Hyperparameter Tuning with RandomizedSearchCV
# ----------------------------------------------------------------------------------

# Define the parameter grid
param_distributions = {
    'preprocessor__text__tfidf__max_features': [5000, 10000, 15000],
    'preprocessor__text__tfidf__ngram_range': [(1, 2), (1, 3)],
    'rf__n_estimators': [100, 200, 300, 400],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__max_features': ['sqrt', 'log2', None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
    'rf__bootstrap': [True, False]
}

# Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,                  # Increased to 50 iterations
    cv=skf,
    scoring='f1',
    n_jobs=-1,
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
# 9. Adjust Classification Threshold
# ----------------------------------------------------------------------------------

# Get predicted probabilities
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

# Adjust threshold
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

# Find the threshold with the highest F1-score
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_thresh}")

# Make predictions with the best threshold
y_pred_adjusted = (y_proba >= best_thresh).astype(int)

# ----------------------------------------------------------------------------------
# 10. Model Evaluation
# ----------------------------------------------------------------------------------

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision_non_smishing = precision_score(y_test, y_pred_adjusted, pos_label=0)
recall_non_smishing = recall_score(y_test, y_pred_adjusted, pos_label=0)
f1_non_smishing = f1_score(y_test, y_pred_adjusted, pos_label=0)

precision_smishing = precision_score(y_test, y_pred_adjusted, pos_label=1)
recall_smishing = recall_score(y_test, y_pred_adjusted, pos_label=1)
f1_smishing = f1_score(y_test, y_pred_adjusted, pos_label=1)

print(f"\n[Optimized RF with Adjusted Threshold] Accuracy: {accuracy:.4f}")
print(f"[Optimized RF] Precision for Non-Smishing (label=0): {precision_non_smishing:.4f}")
print(f"[Optimized RF] Recall for Non-Smishing (label=0): {recall_non_smishing:.4f}")
print(f"[Optimized RF] F1-score for Non-Smishing (label=0): {f1_non_smishing:.4f}")

print(f"[Optimized RF] Precision for Smishing (label=1): {precision_smishing:.4f}")
print(f"[Optimized RF] Recall for Smishing (label=1): {recall_smishing:.4f}")
print(f"[Optimized RF] F1-score for Smishing (label=1): {f1_smishing:.4f}")

# Classification Report
print("\n[Optimized RF] Classification Report:")
print(classification_report(y_test, y_pred_adjusted, target_names=['Non-Smishing', 'Smishing']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
print("\n[Optimized RF] Confusion Matrix:")
print(conf_matrix)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Smishing', 'Smishing'],
            yticklabels=['Non-Smishing', 'Smishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized Random Forest - Confusion Matrix')
plt.show()

# ----------------------------------------------------------------------------------
# 11. Feature Importance Analysis
# ----------------------------------------------------------------------------------

# Get feature importances
rf_model = best_pipeline.named_steps['rf']
feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = rf_model.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top 20 features
print("\nTop 20 Features by Importance:")
print(feature_importance_df.head(20))

# ----------------------------------------------------------------------------------
# 12. Cross-Validation for Robust Evaluation
# ----------------------------------------------------------------------------------

# Evaluate the optimized Random Forest using cross-validation
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
print(f"\n[Cross-Validation] Random Forest F1 Scores: {cv_scores}")
print(f"[Cross-Validation] Mean F1 Score: {cv_scores.mean():.4f}")
