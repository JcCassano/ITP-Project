import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the SMS dataset
sms_data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# 2. Preprocess the data
sms_data['Label'] = sms_data['LABEL'].apply(lambda x: 1 if x.lower() == 'smishing' else 0)

# Extract features and labels
X_sms = sms_data['TEXT'].values  # Feature: SMS content
y_sms = sms_data['Label'].values  # Label: Smishing (1) or Ham (0)

# Check for missing values and replace with an empty string if any
X_sms = [str(x) if pd.notna(x) else '' for x in X_sms]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sms, y_sms, test_size=0.2, random_state=42)

# 4. Tokenize and pad the SMS texts
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='UNK')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input length (maxlen = 100)
X_train_padded = pad_sequences(X_train_seq, maxlen=100)
X_test_padded = pad_sequences(X_test_seq, maxlen=100)


# 5. Build the optimized LSTM model
def build_lstm_model(input_length, vocab_size):
    model = keras.Sequential()

    # Embedding layer (trainable)
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=200,  # Larger embedding size for more expressive representations
                        input_length=input_length,
                        trainable=True))
    model.add(SpatialDropout1D(0.2))

    # Single LSTM layer with 384 units
    model.add(LSTM(384, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))  # More LSTM units

    # Dense layer with Dropout
    model.add(Dense(128, activation='relu'))  # Increased dense units to 128
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Define parameters for the LSTM model
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
input_length = 100  # Length of input sequences

# Build the LSTM model
model = build_lstm_model(input_length, vocab_size)

# 6. Train the model with early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.00001)

model.fit(X_train_padded, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1,
          callbacks=[early_stopping, reduce_lr])

# 7. Predict on the test data
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)

# 8. Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 9. Print results
print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing (label=1): {precision:.4f}")
print(f"Recall for detecting smishing (label=1): {recall:.4f}")
print(f"F1-score for detecting smishing (label=1): {f1:.4f}")