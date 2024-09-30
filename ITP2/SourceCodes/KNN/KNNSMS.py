import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels to integers
label_mapping = {'ham': 0, 'spam': 1, 'smishing': 2}
data['Label'] = data['LABEL'].str.lower().map(label_mapping)

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Define features and target variable
X = data['TEXT']
y = data['Label'].astype(int)

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Transform the text data using TF-IDF vectorization with adjusted parameters
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),       # Include unigrams and bigrams
    max_df=0.9,
    max_features=10000        # Limit vocabulary size
)

X_train_tfidf = vectorizer.fit_transform(X_train_texts)
X_test_tfidf = vectorizer.transform(X_test_texts)

# Dimensionality Reduction using Truncated SVD
svd = TruncatedSVD(n_components=300, random_state=42)  # Adjust n_components as needed
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_reduced = svd.transform(X_test_tfidf)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reduced, y_train)

# Initialize the KNN classifier with optimized parameters
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='cosine',
    n_jobs=-1
)

# Train the classifier
knn.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = knn.predict(X_test_reduced)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Compute precision, recall, and F1-score for the smishing class (label=2)
precision_smishing = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
recall_smishing = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
f1_smishing = f1_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=2): {precision_smishing:.4f}")
print(f"Recall for detecting smishing messages (label=2): {recall_smishing:.4f}")
print(f"F1-score for detecting smishing messages (label=2): {f1_smishing:.4f}")

# # Classification Report
# labels = ['ham', 'spam', 'smishing']
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=labels))
#
# # Confusion Matrix
# print("\nConfusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
#
# # Visualize the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()