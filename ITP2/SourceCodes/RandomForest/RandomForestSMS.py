import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

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

# Map labels to integers
label_mapping = {'ham': 0, 'spam': 1, 'smishing': 2}
data['Label'] = data['LABEL'].str.lower().map(label_mapping)

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Convert labels to integers
data['Label'] = data['Label'].astype(int)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Define features and target variable
X = data['TEXT']
y = data['Label']

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Transform the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

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
# print("\nClassification Report:")
# target_names = ['Ham', 'Spam', 'Smishing']
# print(classification_report(y_test, y_pred, target_names=target_names))
#
# # Confusion Matrix
# print("\nConfusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
#
# # Visualize the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=target_names, yticklabels=target_names)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
