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
file_path = 'C:/Users/Admin/Desktop/ITP2/cleaned_sms.csv'
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

# Check the unique labels
print("Unique labels in the dataset after normalization:", sms_df['LABEL'].unique())

# Split dataset into features (X) and labels (y)
X = sms_df['processed_text']
y = sms_df['LABEL']  # Assuming 'LABEL' is the column that contains spam/ham labels

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

# Print unique labels in the test set
print("Unique labels in the test set:", np.unique(y_test))

# Calculate precision, recall, and F1 score using weighted average for multiclass support
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Show detailed classification report
# Use the unique labels from your dataset for target_names
unique_labels = np.unique(y_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=unique_labels))
