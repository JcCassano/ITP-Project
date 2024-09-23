import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels: 'Phishing Email' to 1, 'Safe Email' to 0
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
data['Email Type'] = data['Email Type'].map(label_mapping)

# Verify label encoding
print("\nLabel distribution:")
print(data['Email Type'].value_counts())

# Define features and target variable
X = data['Email Text'].values
y = data['Email Type'].values

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_texts)
X_test_tfidf = vectorizer.transform(X_test_texts)

# Check class imbalance
print("\nOriginal label distribution:")
print(pd.Series(y_train).value_counts())

# Apply RandomOverSampler to the training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, y_train)

# Shuffle the resampled data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Verify new label distribution
print("\nAfter oversampling:")
print(pd.Series(y_train_resampled).value_counts())

# Initialize the SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear', class_weight=None, probability=True, random_state=42)

# Train the classifier
svm_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_phishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_phishing = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_phishing = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting phishing emails (label=1): {precision_phishing:.4f}")
print(f"Recall for detecting phishing emails (label=1): {recall_phishing:.4f}")
print(f"F1-score for detecting phishing emails (label=1): {f1_phishing:.4f}")

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Phishing Email']))
#
# # Confusion Matrix
# print("\nConfusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
#
# # Visualize the confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Safe Email', 'Phishing Email'],
#             yticklabels=['Safe Email', 'Phishing Email'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
