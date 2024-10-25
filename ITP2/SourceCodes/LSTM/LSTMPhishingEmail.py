import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import nltk for lemmatization and stemming
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required nltk datasets (only need to run once)
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Lemmatizer and Stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# 1. Load the phishing email dataset
email_data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Display the first few rows of the dataset to understand its structure
print("Dataset Structure:\n", email_data.head())

# 2. Preprocess the data
# Convert 'Email Type' to binary labels: 1 for 'Phishing Email', 0 for 'Safe Email'
email_data['Label'] = email_data['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)


# 3. Text Preprocessing: Lemmatization and Stemming
def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Lemmatize and Stem each word in the text
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
    #stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    # Join the processed words back into a single string
    return ' '.join(lemmatized_words)

# Apply the preprocessing to the 'Email Text' column
email_data['Processed_Text'] = email_data['Email Text'].apply(lambda x: preprocess_text(str(x)))

# Extract features and labels
X_email = email_data['Processed_Text'].values  # Feature: Email content
y_email = email_data['Label'].values  # Label: Phishing (1) or Safe (0)

# Check for missing values and replace with an empty string if any
X_email = [str(x) if pd.notna(x) else '' for x in X_email]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_email, y_email, test_size=0.2, random_state=42)

# 4. Tokenize and pad the email texts
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='UNK')  # Tokenizer with a vocabulary size of 5000 words
tokenizer.fit_on_texts(X_train)  # Learn the word index from the training email texts

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input length (maxlen = 100)
X_train_padded = pad_sequences(X_train_seq, maxlen=100)
X_test_padded = pad_sequences(X_test_seq, maxlen=100)

# Load Pretrained GloVe Embeddings (50-dimensional)
embedding_index = {}
glove_file = "../../SourceCodes/LSTM/glove.6B.50d.txt"  # Make sure this file exists in your system
with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coef

# Prepare the embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 5. Build the optimized LSTM model
def build_lstm_model(input_length, vocab_size, embedding_matrix, embedding_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=True))  # Now trainable for fine-tuning

    model.add(SpatialDropout1D(0.2))  # Dropout for regularization

    # Single LSTM layer with more units
    model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))  # Increase LSTM units to 256

    # Add Dense layers for more complexity
    model.add(Dense(64, activation='relu'))  # Extra dense layer with 64 units
    model.add(Dropout(0.3))  # Dropout for dense layer

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build the LSTM model
model = build_lstm_model(input_length=100, vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_dim=embedding_dim)

# 6. Train the model (with ReduceLROnPlateau and EarlyStopping)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

model.fit(X_train_padded, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1, callbacks=[early_stopping, reduce_lr])

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
