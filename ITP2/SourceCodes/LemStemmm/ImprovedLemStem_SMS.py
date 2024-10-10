import pandas as pd
import numpy as np
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer, stemmer, and stopwords
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization and stemming
    processed_tokens = [
        stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in stop_words
    ]
    return ' '.join(processed_tokens)

# Augmentation function: synonym replacement
def augment_text(text, n_augments=1):
    words = text.split()
    augmented_texts = []
    if len(words) == 0:
        return augmented_texts  # Return empty list if there are no words
    for _ in range(n_augments):
        random_word = random.choice(words)
        new_words = [lemmatizer.lemmatize(w) if w != random_word else random_word for w in words]
        augmented_texts.append(' '.join(new_words))
    return augmented_texts

# Load the CSV dataset (update file path if necessary)
file_path = 'C:/Users/Admin/Desktop/ITP2/cleaned_sms.csv'
sms_df = pd.read_csv(file_path)

# Print the columns to verify structure
print("Available columns in the dataset:")
print(sms_df.columns)

# Update to use the correct column names (assuming 'TEXT' contains SMS body and 'LABEL' contains the class)
sms_df['text'] = sms_df['TEXT'].fillna('')

# Apply preprocessing to the text column
sms_df['processed_text'] = sms_df['text'].apply(preprocess_text)

# Normalize the 'LABEL' column to lowercase
sms_df['LABEL'] = sms_df['LABEL'].str.lower()

# Augmenting the dataset
augmented_texts = []
labels = []

for _, row in sms_df.iterrows():
    augmented_samples = augment_text(row['processed_text'], n_augments=2)  # Two augmented versions
    if augmented_samples:  # Check if augmentation generated any text
        augmented_texts.extend(augmented_samples)
        labels.extend([row['LABEL']] * len(augmented_samples))  # Append the same label for augmented texts

# Create a DataFrame for the augmented data
augmented_df = pd.DataFrame({'processed_text': augmented_texts, 'LABEL': labels})

# Combine the original and augmented data
final_df = pd.concat([sms_df[['processed_text', 'LABEL']], augmented_df], ignore_index=True)

# Split dataset into features (X) and labels (y)
X = final_df['processed_text']
y = final_df['LABEL']

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

# Apply SMOTE for balancing the classes
smote = SMOTE(random_state=42)
X_tfidf_res, y_res = smote.fit_resample(X_tfidf, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_res, y_res, test_size=0.2, random_state=42)

# Build a neural network model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Print detailed classification report
unique_labels = np.unique(y_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=unique_labels))
