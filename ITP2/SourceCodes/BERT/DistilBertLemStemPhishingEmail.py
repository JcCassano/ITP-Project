import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
import spacy

# Load spaCy's English language model for lemmatization
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000

# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Load the dataset
data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels: 'Phishing Email' to 1, 'Safe Email' to 0
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
data['Email Type'] = data['Email Type'].map(label_mapping)

# Function to apply lemmatization
def lemmatize_text(text):
    # Split text into chunks of 500,000 characters to avoid memory issues
    chunk_size = 500000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    lemmatized_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        lemmatized_chunks.append(" ".join([token.lemma_ for token in doc]))
    return " ".join(lemmatized_chunks)


# Function to apply stemming
def stem_text(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# Apply lemmatization and stemming to the 'Email Text' column
data['Email Text'] = data['Email Text'].apply(lemmatize_text)
data['Email Text'] = data['Email Text'].apply(stem_text)

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

# Handle class imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_texts_resampled, y_train_resampled = ros.fit_resample(X_train_texts.reshape(-1, 1), y_train)
X_train_texts_resampled = X_train_texts_resampled.flatten()

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the input texts
train_encodings = tokenizer(list(X_train_texts_resampled), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test_texts), truncation=True, padding=True, max_length=512)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train_resampled
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

# Build the model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
batch_size = 8  # Smaller batch size due to larger sequence length
epochs = 3  # Adjust based on your computational resources

history = model.fit(train_dataset.shuffle(1000).batch(batch_size),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=test_dataset.batch(batch_size))

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred_prob = model.predict(test_dataset.batch(batch_size)).logits
y_pred = np.argmax(y_pred_prob, axis=1)

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
