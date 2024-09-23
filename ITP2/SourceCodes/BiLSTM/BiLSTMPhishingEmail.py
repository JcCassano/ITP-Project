import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels: 'Phishing Email' to 1, 'Safe Email' to 0
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
data['Email Type'] = data['Email Type'].map(label_mapping)

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

# Tokenization
max_words = 20000  # Limit vocabulary size to the top 20,000 words
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_texts)

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)

# Set a fixed maximum sequence length
max_seq_length = 1000  # Adjust based on your dataset and memory constraints

# Pad sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_seq_length, padding='post', truncating='post')

# Check class imbalance
print("\nOriginal label distribution:")
print(pd.Series(y_train).value_counts())

# Apply RandomOverSampler to the training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_padded, y_train)

# Shuffle the resampled data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Verify new label distribution
print("\nAfter oversampling:")
print(pd.Series(y_train_resampled).value_counts())

# Build the Bidirectional LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words,  # Use max_words instead of len(tokenizer.word_index) + 1
                              output_dim=128,
                              input_length=max_seq_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
batch_size = 32
epochs = 10  # Increase epochs for better learning

history = model.fit(X_train_resampled, y_train_resampled,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test_padded, y_test))

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred_prob = model.predict(X_test_padded)
y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_phishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_phishing = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_phishing = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting phishing emails (label=1): {precision_phishing:.4f}")
print(f"Recall for detecting phishing emails (label=1): {recall_phishing:.4f}")
print(f"F1-score for detecting phishing emails (label=1): {f1_phishing:.4f}")
