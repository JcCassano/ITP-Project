import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    text = ' '.join(filtered_tokens)
    return text

# Apply preprocessing to the 'Email Text' column
data['Cleaned Email Text'] = data['Email Text'].apply(preprocess_text)

# Define features and target variable
X = data['Cleaned Email Text'].values
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
max_seq_length = 500  # Reduced for faster computation and to focus on important parts

# Pad sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_seq_length, padding='post', truncating='post')

# Check class imbalance
print("\nOriginal label distribution:")
print(pd.Series(y_train).value_counts())

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))


embedding_dim = 300  # Increased embedding dimensions

model = Sequential([
    Embedding(input_dim=max_words,
              output_dim=embedding_dim,
              input_length=max_seq_length,
              trainable=True),  # Set trainable to True
    Dropout(0.5),  # Add dropout to embedding layer
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),  # Add batch normalization
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Add L2 regularization
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Lowered learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Print model summary
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

batch_size = 32
epochs = 20  # Increased number of epochs

history = model.fit(X_train_padded, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test_padded, y_test),
                    class_weight=class_weight_dict,
                    callbacks=[early_stopping])


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
