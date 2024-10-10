import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    # Join tokens back into a string
    return ' '.join(processed_tokens)

# Load the CSV dataset
file_path = 'C:/Users/Admin/Desktop/ITP2/Phishing_Email.csv'
emails_df = pd.read_csv(file_path)

# Update to use the correct column names
emails_df['text'] = emails_df['Email Text'].fillna('')

# Apply the preprocessing function to the text
emails_df['processed_text'] = emails_df['text'].apply(preprocess_text)

# Split dataset into features (X) and labels (y)
X = emails_df['processed_text']
y = emails_df['Email Type']

# Vectorize the text using TF-IDF (increase features, use trigrams)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define individual models
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                    max_iter=500, random_state=42, learning_rate_init=0.001, alpha=0.001)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Combine models using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('mlp', mlp),
    ('rf', random_forest),
    ('gb', gradient_boosting)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred = ensemble_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, pos_label='Phishing Email')
recall = recall_score(y_test, y_pred, pos_label='Phishing Email')
f1 = f1_score(y_test, y_pred, pos_label='Phishing Email')

# Print the results
print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Show detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Phishing Email', 'Safe Email']))

# Optional: Fine-tune hyperparameters using GridSearchCV for MLPClassifier
param_grid = {
    'hidden_layer_sizes': [(100,), (128, 64), (150, 100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.0001]
}

grid_search = GridSearchCV(MLPClassifier(max_iter=500, random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

# Test the best model found by GridSearchCV
best_model = grid_search.best_estimator_
y_best_pred = best_model.predict(X_test)

# Print final accuracy of the best model
accuracy_best = accuracy_score(y_test, y_best_pred)
print(f'Accuracy of Best Model: {accuracy_best * 100:.4f}%')
