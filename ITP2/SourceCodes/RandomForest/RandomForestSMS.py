import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values with empty strings (if any)
data = data.fillna('')

# Map labels: 'smishing' to 1, others ('ham', 'spam') to 0
data['Label'] = data['LABEL'].apply(lambda x: 1 if x.lower() == 'smishing' else 0)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Define features and target variable
X = data['TEXT']
y = data['Label']

# Optional: Include other attributes if desired
# Uncomment the following lines to include 'URL', 'EMAIL', 'PHONE' as features
# X = data[['TEXT', 'URL', 'EMAIL', 'PHONE']]

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Transform the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_smishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=1): {precision_smishing:.4f}")

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Non-Smishing', 'Smishing']))
#
# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(conf_matrix)
#
# # Visualize the confusion matrix
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Non-Smishing', 'Smishing'],
#             yticklabels=['Non-Smishing', 'Smishing'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()