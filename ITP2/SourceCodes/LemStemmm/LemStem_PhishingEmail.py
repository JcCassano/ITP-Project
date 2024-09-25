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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')  # Add this line to download the punkt_tab resource

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

# Load the CSV dataset
file_path = 'C:/Users/Admin/Desktop/ITP2/Phishing_Email.csv'
emails_df = pd.read_csv(file_path)

# Print the columns to verify structure
print("Available columns in the dataset:")
print(emails_df.columns)

# Update to use the correct column names
emails_df['text'] = emails_df['Email Text'].fillna('')  # Assuming 'Email Text' contains the email body

# Apply the preprocessing function to the combined text field
emails_df['processed_text'] = emails_df['text'].apply(preprocess_text)

# Split dataset into features (X) and labels (y)
X = emails_df['processed_text']
y = emails_df['Email Type']  # Assuming 'Email Type' is the label (phishing or not)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build a neural network model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, pos_label='Phishing Email')  # Use 'Phishing Email'
recall = recall_score(y_test, y_pred, pos_label='Phishing Email')
f1 = f1_score(y_test, y_pred, pos_label='Phishing Email')

# Print the results
print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Show detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Phishing Email', 'Safe Email']))
