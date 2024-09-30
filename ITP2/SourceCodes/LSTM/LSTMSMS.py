import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the SMS dataset
sms_data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Display the first few rows of the dataset to understand its structure
print("Dataset Structure:\n", sms_data.head())

# 2. Preprocess the data
# Convert 'LABEL' to binary labels: 1 for 'Smishing', 0 for 'ham'
sms_data['Label'] = sms_data['LABEL'].apply(lambda x: 1 if x.lower() == 'smishing' else 0)

# Display the first few rows of the dataset to understand its structure
#print("Dataset Structure:\n", sms_data.head())

# Extract features and labels
X_sms = sms_data['TEXT'].values  # Feature: SMS content
y_sms = sms_data['Label'].values  # Label: Smishing (1) or Ham (0)

# Check for missing values and replace with an empty string if any
X_sms = [str(x) if pd.notna(x) else '' for x in X_sms]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sms, y_sms, test_size=0.2, random_state=42)

# 4. Tokenize and pad the SMS texts
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='UNK')  # Tokenizer with a vocabulary size of 5000 words
tokenizer.fit_on_texts(X_train)  # Learn the word index from the training SMS texts

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input length (maxlen = 100)
X_train_padded = pad_sequences(X_train_seq, maxlen=100)
X_test_padded = pad_sequences(X_test_seq, maxlen=100)

# 5. Build the LSTM model
def build_lstm_model(input_length, vocab_size):
    model = keras.Sequential()  # Initialize a Sequential model
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=input_length))  # Embedding layer
    model.add(SpatialDropout1D(0.2))  # Apply dropout for regularization
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer with dropout
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile the model
    return model

# Define parameters for the LSTM model
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (number of unique tokens in the dataset)
input_length = 100  # Length of input sequences (same as maxlen used in padding)

# Build the LSTM model for SMS smishing detection
model = build_lstm_model(input_length, vocab_size)

# 6. Train the model
model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# 7. Predict on the test data
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)  # Convert probabilities to binary predictions

# 8. Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 9. Print results
print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting phishing emails (label=1): {precision:.4f}")
print(f"Recall for detecting phishing emails (label=1): {recall:.4f}")
print(f"F1-score for detecting phishing emails (label=1): {f1:.4f}")