import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels to integers
label_mapping = {'ham': 0, 'spam': 1, 'smishing': 2}
data['Label'] = data['LABEL'].str.lower().map(label_mapping)

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Convert labels to integers
data['Label'] = data['Label'].astype(int)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Define features and target variable
X = data['TEXT'].values
y = data['Label'].values

# Split the dataset
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)

# Set maximum sequence length
max_seq_length = 300

# Pad sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_seq_length, padding='post')

# Apply RandomOverSampler to the training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_padded, y_train)

# Shuffle the resampled data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_dict = dict(enumerate(class_weights))

# Verify new label distribution
print("\nAfter oversampling:")
print(pd.Series(y_train_resampled).value_counts())

# Build the model with LSTM and GRU hybrid architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                              output_dim=200,  # Increased embedding dimension
                              input_length=max_seq_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.4)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, dropout=0.4)),  # Replaced one LSTM with GRU
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))
])

# Compile the model with a slightly reduced learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model with early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

batch_size = 64
epochs = 20

history = model.fit(X_train_resampled, y_train_resampled,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test_padded, y_test),
                    class_weight=class_weights_dict,  # Added class weighting
                    callbacks=[early_stopping, lr_scheduler])

# Make predictions
y_pred_prob = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_smishing = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
recall_smishing = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
f1_smishing = f1_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=2): {precision_smishing:.4f}")
print(f"Recall for detecting smishing messages (label=2): {recall_smishing:.4f}")
print(f"F1-score for detecting smishing messages (label=2): {f1_smishing:.4f}")
