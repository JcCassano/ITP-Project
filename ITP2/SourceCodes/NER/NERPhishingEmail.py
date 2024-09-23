import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Load the dataset
data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Add a new column for email length
data['Email Length'] = data['Email Text'].apply(len)

# Define the maximum length
MAX_LENGTH = 10000  # (e.g., 10,000 characters), more memory = can increase more length

# Option 1: Remove emails exceeding the maximum length
data = data[data['Email Length'] <= MAX_LENGTH]

# Option 2: Truncate emails exceeding the maximum length
# data['Email Text'] = data['Email Text'].apply(lambda x: x[:MAX_LENGTH] if len(x) > MAX_LENGTH else x)

# Drop the 'Email Length' column if not needed
data = data.drop(columns=['Email Length'])

# Map labels: 'Phishing Email' to 1, 'Safe Email' to 0
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
data['Email Type'] = data['Email Type'].map(label_mapping)

# Verify label encoding
print("\nLabel distribution:")
print(data['Email Type'].value_counts())

# Define a function to extract relevant entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = {
        'PERSON': 0,
        'ORG': 0,
        'MONEY': 0,
        'DATE': 0,
        'GPE': 0,
        'CARDINAL': 0,
        'PERCENT': 0,
        'QUANTITY': 0
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] += 1
    return pd.Series(entities)

# Apply the function to extract entities from each email
print("\nExtracting entities from emails...")
entity_features = data['Email Text'].apply(extract_entities)

# Combine the original data with the entity features
data = pd.concat([data.reset_index(drop=True), entity_features.reset_index(drop=True)], axis=1)

# Feature Engineering: Additional textual features
data['email_length'] = data['Email Text'].apply(len)
data['num_exclamations'] = data['Email Text'].str.count('!')
data['num_questions'] = data['Email Text'].str.count('\?')
data['num_uppercase'] = data['Email Text'].apply(lambda x: sum(1 for c in x if c.isupper()))
data['num_digits'] = data['Email Text'].str.count('\d')

# Define the feature set and target variable
X = data[['PERSON', 'ORG', 'MONEY', 'DATE', 'GPE', 'CARDINAL', 'PERCENT', 'QUANTITY',
          'email_length', 'num_exclamations', 'num_questions', 'num_uppercase', 'num_digits']]
y = data['Email Type']

# Transform the email text using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf_features = vectorizer.fit_transform(data['Email Text'])

# Combine NER features and TF-IDF features
ner_features = data[['PERSON', 'ORG', 'MONEY', 'DATE', 'GPE', 'CARDINAL', 'PERCENT', 'QUANTITY',
                     'email_length', 'num_exclamations', 'num_questions', 'num_uppercase', 'num_digits']]

# Convert NER features to a sparse matrix
ner_features_sparse = csr_matrix(ner_features.values)

# Combine the features
X_combined = hstack([tfidf_features, ner_features_sparse])

# Proceed with train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.25, random_state=42, stratify=y
)

# Initialize and train the Logistic Regression classifier
classifier = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision_phishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the combined model: {accuracy:.4f}")
print(f"Precision for detecting phishing emails (label=1): {precision_phishing:.4f}")

# (Optional) Proceed with further evaluation and visualization


# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Phishing Email']))
#
# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(conf_matrix)
#
# # Visualize the confusion matrix
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Safe Email', 'Phishing Email'],
#             yticklabels=['Safe Email', 'Phishing Email'])
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
# print("\nTop Features Contributing to Phishing Detection:")
# print(importance_df.head(10))
#
# # Plot feature importances
# importance_df.head(10).plot.bar(x='Feature', y='Coefficient', figsize=(12,6))
# plt.title('Top Features Contributing to Phishing Detection')
# plt.xlabel('Features')
# plt.ylabel('Coefficient')
# plt.tight_layout()
# plt.show()
