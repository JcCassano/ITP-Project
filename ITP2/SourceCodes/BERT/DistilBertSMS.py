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

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels: 'smishing' to 1, others ('ham', 'spam') to 0
data['Label'] = data['LABEL'].apply(lambda x: 1 if x.lower() == 'smishing' else 0)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Define features and target variable
X = data['TEXT'].values
y = data['Label'].values

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
train_encodings = tokenizer(list(X_train_texts_resampled), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test_texts), truncation=True, padding=True, max_length=128)

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
precision_smishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_smishing = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_smishing = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=1): {precision_smishing:.4f}")
print(f"Recall for detecting smishing messages (label=1): {recall_smishing:.4f}")
print(f"F1-score for detecting smishing messages (label=1): {f1_smishing:.4f}")

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Non-Smishing', 'Smishing']))
#
# # Confusion Matrix
# print("\nConfusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
#
# # Visualize the confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Non-Smishing', 'Smishing'],
#             yticklabels=['Non-Smishing', 'Smishing'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
