import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')
# Remember to run python -m spacy download en_core_web_sm to run the model

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values with empty strings
data = data.fillna('')

# Map labels: 'smishing' to 1, others ('ham', 'spam') to 0
data['Label'] = data['LABEL'].apply(lambda x: 1 if x.lower() == 'smishing' else 0)

# Verify label encoding
print("\nLabel distribution:")
print(data['Label'].value_counts())

# Define a function to extract relevant entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = {
        'URL': 0,
        'EMAIL': 0,
        'PHONE': 0,
        'PERSON': 0,
        'ORG': 0,
        'MONEY': 0,
        'DATE': 0,
        'TIME': 0,
        'GPE': 0,
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] += 1
    return pd.Series(entities)

# Apply the function to extract entities from each message
print("\nExtracting entities from messages...")
entity_features = data['TEXT'].apply(extract_entities)

# Combine the original data with the entity features
data = pd.concat([data, entity_features], axis=1)

# Feature Engineering: Additional textual features
data['message_length'] = data['TEXT'].apply(len)
data['num_exclamations'] = data['TEXT'].str.count('!')
data['num_questions'] = data['TEXT'].str.count('\?')
data['num_uppercase'] = data['TEXT'].apply(lambda x: sum(1 for c in x if c.isupper()))
data['num_digits'] = data['TEXT'].str.count('\d')

# Define the feature set and target variable
X = data[['PERSON', 'ORG', 'MONEY', 'DATE', 'TIME', 'GPE',
          'message_length', 'num_exclamations', 'num_questions',
          'num_uppercase', 'num_digits']]

y = data['Label']

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Initialize the Logistic Regression classifier
classifier = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_smishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_smishing = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=1): {precision_smishing:.4f}")
print(f"Recall for detecting smishing messages (label=1): {recall_smishing:.4f}")

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
#
# # Feature Importance Analysis
# feature_names = X.columns
# coefficients = classifier.coef_[0]
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Coefficient': coefficients
# }).sort_values(by='Coefficient', ascending=False)
#
# # Display top features
# print("\nTop Features Contributing to Smishing Detection:")
# print(importance_df)
#
# # Plot feature importances
# importance_df.plot.bar(x='Feature', y='Coefficient', figsize=(12,6))
# plt.title('Feature Importance in Smishing Detection')
# plt.xlabel('Features')
# plt.ylabel('Coefficient')
# plt.tight_layout()
# plt.show()
