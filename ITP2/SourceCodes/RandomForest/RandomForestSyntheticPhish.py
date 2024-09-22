# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../Dataset/synthetic_phish.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values with empty strings (if any)
data = data.fillna('')

# Define features and target variable
X = data['text']
y = data['label']

# Check the distribution of labels
print("\nLabel distribution:")
print(y.value_counts())

# Split the dataset into training and testing sets (75% training, 25% testing) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Verify class distribution in training and test sets
print("\nTraining set label distribution:")
print(y_train.value_counts())
print("\nTest set label distribution:")
print(y_test.value_counts())

# Transform the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Include additional features (e.g., email length)
def extract_additional_features(text_series):
    features = pd.DataFrame()
    features['email_length'] = text_series.apply(len)
    features['num_exclamations'] = text_series.str.count('!')
    features['num_questions'] = text_series.str.count('\?')
    features['num_uppercase'] = text_series.apply(lambda x: sum(1 for c in x if c.isupper()))
    return features

X_train_additional = extract_additional_features(X_train)
X_test_additional = extract_additional_features(X_test)

# Combine TF-IDF features with additional features
X_train_combined = hstack([X_train_tfidf, X_train_additional])
X_test_combined = hstack([X_test_tfidf, X_test_additional])

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

# Train the classifier on the balanced dataset
classifier.fit(X_train_balanced, y_train_balanced)

# Make predictions on the test set
y_pred = classifier.predict(X_test_combined)
y_pred_proba = classifier.predict_proba(X_test_combined)[:, 1]  # Probability for class 1

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_phishing = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
recall_phishing = recall_score(y_test, y_pred, pos_label=0, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting phishing emails (label=0): {precision_phishing:.4f}")
print(f"Recall for detecting phishing emails (label=0): {recall_phishing:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Phishing', 'Non-Phishing'],
            yticklabels=['Phishing', 'Non-Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Adjust the classification threshold
threshold = 0.4  # Adjust as needed
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Recalculate metrics with adjusted threshold
accuracy_adj = accuracy_score(y_test, y_pred_adjusted)
precision_phishing_adj = precision_score(y_test, y_pred_adjusted, pos_label=0, zero_division=0)
recall_phishing_adj = recall_score(y_test, y_pred_adjusted, pos_label=0, zero_division=0)

print(f"\nAdjusted Threshold ({threshold}) Results:")
print(f"Accuracy: {accuracy_adj:.4f}")
print(f"Precision for Phishing Emails: {precision_phishing_adj:.4f}")
print(f"Recall for Phishing Emails: {recall_phishing_adj:.4f}")

# Confusion Matrix with adjusted threshold
conf_matrix_adj = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix with Adjusted Threshold:")
print(conf_matrix_adj)

# Visualize the adjusted confusion matrix
sns.heatmap(conf_matrix_adj, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Phishing', 'Non-Phishing'],
            yticklabels=['Phishing', 'Non-Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix with Adjusted Threshold ({threshold})')
plt.show()

# Analyze feature importance
feature_importances = classifier.feature_importances_
feature_names = vectorizer.get_feature_names_out()
additional_feature_names = X_train_additional.columns.tolist()
all_feature_names = np.concatenate((feature_names, additional_feature_names))

importances_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

# Display top 20 features
print("\nTop 20 Important Features:")
print(importances_df.head(20))

# Plot feature importances
importances_df.head(20).plot.bar(x='feature', y='importance', figsize=(12,6))
plt.title('Top 20 Important Features')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Perform cross-validation
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(
    classifier, X_train_balanced, y_train_balanced, cv=skf, scoring='recall', n_jobs=-1
)
print(f"\nCross-Validation Recall Scores: {cv_scores}")
print(f"Mean Cross-Validation Recall: {cv_scores.mean():.4f}")
