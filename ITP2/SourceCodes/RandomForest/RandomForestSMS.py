import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------
# 1. Data Loading and Initial Exploration
# ----------------------------------------------------------------------------------

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

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

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# ----------------------------------------------------------------------------------
# 3. Train-Test Split
# ----------------------------------------------------------------------------------

# Define features and target variable
X = data['TEXT']
y = data['Label']

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------------
# 4. Building the Pipeline
# ----------------------------------------------------------------------------------

# Initialize TF-IDF Vectorizer with optimized parameters
tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.9,
    min_df=5,              # Remove very rare words
    ngram_range=(1, 2),    # Include unigrams and bigrams
    max_features=10000     # Limit to top 10,000 features
)

# Initialize Random Forest Classifier with class_weight='balanced'
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# Create a Pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('rf', rf)
])

# ----------------------------------------------------------------------------------
# 5. Hyperparameter Tuning with GridSearchCV
# ----------------------------------------------------------------------------------

# Define the parameter grid for Random Forest
param_grid = {
    'rf__n_estimators': [200, 300, 400],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                     # 5-fold cross-validation
    scoring='f1',             # Optimize for F1-score (binary)
    n_jobs=-1,                # Use all available cores
    verbose=2
)

# Fit GridSearchCV to the training data
print("\nStarting Grid Search for Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# Best parameters
print("\nBest parameters found: ", grid_search.best_params_)

# ----------------------------------------------------------------------------------
# 6. Training the Optimized Random Forest Model
# ----------------------------------------------------------------------------------

# Use the best estimator from GridSearchCV
best_pipeline = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_pipeline.predict(X_test)

# ----------------------------------------------------------------------------------
# 7. Model Evaluation
# ----------------------------------------------------------------------------------

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision_non_smishing = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
recall_non_smishing = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_non_smishing = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

precision_smishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_smishing = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_smishing = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\n[Optimized RF] Accuracy: {accuracy:.4f}")
print(f"[Optimized RF] Precision for Non-Smishing (label=0): {precision_non_smishing:.4f}")
print(f"[Optimized RF] Recall for Non-Smishing (label=0): {recall_non_smishing:.4f}")
print(f"[Optimized RF] F1-score for Non-Smishing (label=0): {f1_non_smishing:.4f}")

print(f"[Optimized RF] Precision for Smishing (label=1): {precision_smishing:.4f}")
print(f"[Optimized RF] Recall for Smishing (label=1): {recall_smishing:.4f}")
print(f"[Optimized RF] F1-score for Smishing (label=1): {f1_smishing:.4f}")

# Classification Report
print("\n[Optimized RF] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Smishing', 'Smishing']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
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
# 9. Cross-Validation for Robust Evaluation
# ----------------------------------------------------------------------------------

# Define Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the optimized Random Forest using cross-validation
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
print(f"\n[Cross-Validation] Random Forest F1 Scores: {cv_scores}")
print(f"[Cross-Validation] Mean F1 Score: {cv_scores.mean():.4f}")
