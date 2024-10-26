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
from nltk.corpus import wordnet
import spacy

nltk.download('wordnet')
nltk.download('punkt')

# Initialize spaCy's lemmatizer
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

# Function to perform lemmatization
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Function to perform stemming
def stem_text(text):
    words = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in words])

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels to integers
label_mapping = {'ham': 0, 'spam': 1, 'smishing': 2}
data['Label'] = data['LABEL'].str.lower().map(label_mapping)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Apply lemmatization and stemming
data['TEXT'] = data['TEXT'].apply(lemmatize_text)
data['TEXT'] = data['TEXT'].apply(stem_text)

# Define features and target variable
X = data['TEXT'].values
y = data['Label'].values.astype(int)

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
max_length = 128  # You can adjust this value based on your hardware capabilities
train_encodings = tokenizer(list(X_train_texts_resampled), truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(list(X_test_texts), truncation=True, padding=True, max_length=max_length)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train_resampled
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

# Build the model with num_labels=3 for multi-class classification
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Define optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
batch_size = 16
epochs = 3  # Adjust based on your computational resources

history = model.fit(train_dataset.shuffle(1000).batch(batch_size),
                    epochs=epochs,
                    validation_data=test_dataset.batch(batch_size))

# Plot training and validation accuracy
# plt.figure(figsize=(12, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# Make predictions on the test set
y_pred_logits = model.predict(test_dataset.batch(batch_size)).logits
y_pred = np.argmax(y_pred_logits, axis=1)

# Compute overall accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute precision, recall, and F1-score for the smishing class (label=2)
precision_smishing = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
recall_smishing = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
f1_smishing = f1_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=2): {precision_smishing:.4f}")
print(f"Recall for detecting smishing messages (label=2): {recall_smishing:.4f}")
print(f"F1-score for detecting smishing messages (label=2): {f1_smishing:.4f}")

# # Map integer labels back to original labels for reporting
# labels = ['ham', 'spam', 'smishing']
#
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
