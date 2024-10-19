import pandas as pd
import numpy as np
import tensorflow as tf

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle

# Import NLTK libraries for tokenization
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files
nltk.download('punkt')

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Merge 'ham' and 'spam' into one category 'non-smishing'
data['LABEL'] = data['LABEL'].str.lower().map({
    'ham': 'non-smishing',
    'spam': 'non-smishing',
    'smishing': 'smishing'
})

# Map the new labels to integers: 'non-smishing' -> 0, 'smishing' -> 1
label_mapping = {'non-smishing': 0, 'smishing': 1}
data['Label'] = data['LABEL'].map(label_mapping)

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Convert labels to integers
data['Label'] = data['Label'].astype(int)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Minimal text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text using NLTK
    tokens = word_tokenize(text)
    # Rejoin tokens into a string
    return ' '.join(tokens)

# Apply preprocessing to the 'TEXT' column
data['Processed_Text'] = data['TEXT'].apply(preprocess_text)

# Define features and target variable
X = data['Processed_Text'].values
y = data['Label'].values

# Split the dataset
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_texts)

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)

# Set maximum sequence length
max_seq_length = 300

# Pad sequences
X_train_padded = pad_sequences(
    X_train_sequences, maxlen=max_seq_length, padding='post', truncating='post'
)
X_test_padded = pad_sequences(
    X_test_sequences, maxlen=max_seq_length, padding='post', truncating='post'
)

# Apply RandomOverSampler to the training data
ros = RandomOverSampler(random_state=42)
X_train_padded_resampled, y_train_resampled = ros.fit_resample(X_train_padded, y_train)

# Shuffle the resampled data
X_train_padded_resampled, y_train_resampled = shuffle(
    X_train_padded_resampled, y_train_resampled, random_state=42
)

# Build a binary classification model (BiLSTM)
def create_bilstm_binary_model(vocab_size, embedding_dim=200, input_length=max_seq_length):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=input_length
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Single output with sigmoid for binary classification
    ])
    return model

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token

# Create the model
bilstm_model = create_bilstm_binary_model(vocab_size)

# Compile the model
bilstm_model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Print model summary
bilstm_model.summary()

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
)

batch_size = 64
epochs = 20

# Train the model
history = bilstm_model.fit(
    X_train_padded_resampled, y_train_resampled,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test_padded, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Make predictions
y_pred_prob = bilstm_model.predict(X_test_padded)
y_pred = np.round(y_pred_prob).astype(int).flatten()  # Convert probabilities to class labels (0 or 1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy of the BiLSTM model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages: {precision:.4f}")
print(f"Recall for detecting smishing messages: {recall:.4f}")
print(f"F1-score for detecting smishing messages: {f1:.4f}")
