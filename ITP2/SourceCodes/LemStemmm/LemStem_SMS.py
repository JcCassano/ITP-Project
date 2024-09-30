import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization and stemming
    processed_tokens = [
        stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in stop_words
    ]
    # Join tokens back into a string
    return ' '.join(processed_tokens)

# Load the CSV dataset (update the file path if needed)
file_path = '../../Dataset/cleaned_sms.csv'
sms_df = pd.read_csv(file_path)

# Print the columns to verify structure
print("Available columns in the dataset:")
print(sms_df.columns)

# Update to use the correct column names
sms_df['text'] = sms_df['TEXT'].fillna('')  # Assuming 'TEXT' contains the SMS body

# Apply the preprocessing function to the text column
sms_df['processed_text'] = sms_df['text'].apply(preprocess_text)

# Normalize the 'LABEL' column to lowercase to avoid duplicates
sms_df['LABEL'] = sms_df['LABEL'].str.lower()

# Map labels to integers
label_mapping = {'ham': 0, 'spam': 1, 'smishing': 2}
sms_df['Label'] = sms_df['LABEL'].map(label_mapping)

# Remove any rows with missing labels after mapping
sms_df = sms_df.dropna(subset=['Label'])

# Convert labels to integers
sms_df['Label'] = sms_df['Label'].astype(int)

# Verify label encoding
print("\nLabel distribution:")
print(sms_df['Label'].value_counts())

# Split dataset into features (X) and labels (y)
X = sms_df['processed_text']
y = sms_df['Label']

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Build a neural network model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute precision, recall, and F1-score for the smishing class (label=2)
precision_smishing = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
recall_smishing = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
f1_smishing = f1_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)

print(f"\nAccuracy of the model: {accuracy * 100:.4f}%")
print(f"Precision for detecting smishing messages (label=2): {precision_smishing:.4f}")
print(f"Recall for detecting smishing messages (label=2): {recall_smishing:.4f}")
print(f"F1-score for detecting smishing messages (label=2): {f1_smishing:.4f}")

# # Classification Report
# target_names = ['ham', 'spam', 'smishing']
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
#
# # Confusion Matrix
# print("\nConfusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
#
# # Optional: Visualize the confusion matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=target_names, yticklabels=target_names)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
