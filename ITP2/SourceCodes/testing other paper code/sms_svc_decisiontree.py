# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------
# 1. Data Loading and Initial Exploration
# ----------------------------------------------------------------------------------

# Load the dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')

# Fill missing values with empty strings (if any)
data = data.fillna('')

# ----------------------------------------------------------------------------------
# 2. Label Mapping and Encoding
# ----------------------------------------------------------------------------------

# Merge 'ham' and 'spam' into 'non-smishing' and map labels to integers
label_mapping = {'ham': 0, 'spam': 0, 'smishing': 1}
data['Label'] = data['LABEL'].str.lower().map(label_mapping)

# Remove any rows with missing labels after mapping
data = data.dropna(subset=['Label'])

# Convert labels to integers
data['Label'] = data['Label'].astype(int)


# ----------------------------------------------------------------------------------
# 3. Feature Engineering for Domain-Based Phishing Detection
# ----------------------------------------------------------------------------------

# Define functions to extract additional features
def has_url(text):
    return int(bool(re.search(r'http\S+|www\.\S+', text)))


def has_phone(text):
    return int(bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}', text)))


def message_length(text):
    return len(text)


def num_exclamations(text):
    return text.count('!')


def num_questions(text):
    return text.count('?')


def num_uppercase(text):
    return sum(1 for c in text if c.isupper())


def num_digits(text):
    return sum(c.isdigit() for c in text)


def num_words(text):
    return len(text.split())


def avg_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words]) if words else 0


def num_special_chars(text):
    return sum(not c.isalnum() and not c.isspace() for c in text)


# Apply feature extraction
data['has_url'] = data['TEXT'].apply(has_url)
data['has_phone'] = data['TEXT'].apply(has_phone)
data['message_length'] = data['TEXT'].apply(message_length)
data['num_exclamations'] = data['TEXT'].apply(num_exclamations)
data['num_questions'] = data['TEXT'].apply(num_questions)
data['num_uppercase'] = data['TEXT'].apply(num_uppercase)
data['num_digits'] = data['TEXT'].apply(num_digits)
data['num_words'] = data['TEXT'].apply(num_words)
data['avg_word_length'] = data['TEXT'].apply(avg_word_length)
data['num_special_chars'] = data['TEXT'].apply(num_special_chars)

# ----------------------------------------------------------------------------------
# 4. Advanced Text Preprocessing for SVC Model
# ----------------------------------------------------------------------------------

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK data files (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


data['TEXT'] = data['TEXT'].apply(preprocess_text)

# ----------------------------------------------------------------------------------
# 5. Define Features and Target Variable
# ----------------------------------------------------------------------------------

# Domain features
domain_features = ['has_url', 'has_phone', 'message_length', 'num_exclamations', 'num_questions',
                   'num_uppercase', 'num_digits', 'num_words', 'avg_word_length', 'num_special_chars']

X_domain = data[domain_features]

# Text feature
X_text = data['TEXT']

# Target variable
y = data['Label']

# ----------------------------------------------------------------------------------
# 6. Split the Dataset into Training and Testing Sets
# ----------------------------------------------------------------------------------

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train_domain, X_test_domain, y_train, y_test = train_test_split(
    X_domain, y, test_size=0.25, random_state=42, stratify=y
)

X_train_text, X_test_text, _, _ = train_test_split(
    X_text, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------------
# 7. Vectorize Text Data
# ----------------------------------------------------------------------------------

# TfidfVectorizer with advanced settings
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),  # Include unigrams, bigrams, trigrams
    max_features=10000,
    sublinear_tf=True
)

X_train_text_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_text_tfidf = tfidf_vectorizer.transform(X_test_text)

# ----------------------------------------------------------------------------------
# 8. Combine Domain and Text Features
# ----------------------------------------------------------------------------------

# Scale domain features
scaler = StandardScaler()
X_train_domain_scaled = scaler.fit_transform(X_train_domain)
X_test_domain_scaled = scaler.transform(X_test_domain)

# Convert domain features to sparse matrix
X_train_domain_sparse = sparse.csr_matrix(X_train_domain_scaled)
X_test_domain_sparse = sparse.csr_matrix(X_test_domain_scaled)

# Combine text and domain features
X_train_combined = sparse.hstack([X_train_text_tfidf, X_train_domain_sparse])
X_test_combined = sparse.hstack([X_test_text_tfidf, X_test_domain_sparse])

# ----------------------------------------------------------------------------------
# 9. Handle Class Imbalance
# ----------------------------------------------------------------------------------

from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_train_combined_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_combined, y_train)

# ----------------------------------------------------------------------------------
# 10. Hyperparameter Tuning for SVC and Decision Tree
# ----------------------------------------------------------------------------------

# Define parameter grid for SVC
svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Define parameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize models
svc = SVC(probability=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

# Randomized Search CV for SVC
svc_random_search = RandomizedSearchCV(
    estimator=svc,
    param_distributions=svc_param_grid,
    n_iter=10,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

svc_random_search.fit(X_train_combined_resampled, y_train_resampled)
best_svc = svc_random_search.best_estimator_

print("Best SVC Parameters:", svc_random_search.best_params_)

# Randomized Search CV for Decision Tree
dt_random_search = RandomizedSearchCV(
    estimator=dt,
    param_distributions=dt_param_grid,
    n_iter=10,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

dt_random_search.fit(X_train_combined_resampled, y_train_resampled)
best_dt = dt_random_search.best_estimator_

print("Best Decision Tree Parameters:", dt_random_search.best_params_)

# ----------------------------------------------------------------------------------
# 11. Ensemble Learning with StackingClassifier
# ----------------------------------------------------------------------------------

# Define base learners
base_learners = [
    ('svc', best_svc),
    ('dt', best_dt)
]

# Meta-classifier
meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_classifier,
    cv=5,
    n_jobs=-1,
    passthrough=False  # Set to True if you want to include original features in meta-classifier
)

# Fit the stacking classifier
stacking_clf.fit(X_train_combined_resampled, y_train_resampled)

# ----------------------------------------------------------------------------------
# 12. Evaluate the Stacking Classifier
# ----------------------------------------------------------------------------------

# Predict on the test set
y_pred_stacking = stacking_clf.predict(X_test_combined)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_stacking)
precision = precision_score(y_test, y_pred_stacking)
recall = recall_score(y_test, y_pred_stacking)
f1 = f1_score(y_test, y_pred_stacking)

print(f"\n[Stacking Classifier] Accuracy: {accuracy:.4f}")
print(f"[Stacking Classifier] Precision: {precision:.4f}")
print(f"[Stacking Classifier] Recall: {recall:.4f}")
print(f"[Stacking Classifier] F1-score: {f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_stacking))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_stacking)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Stacking Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ----------------------------------------------------------------------------------
# 13. Cross-Validation Scores
# ----------------------------------------------------------------------------------

# Cross-validation on the combined training set
cv_scores = cross_val_score(
    stacking_clf,
    X_train_combined_resampled,
    y_train_resampled,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

print("Cross-validated F1 scores:", cv_scores)
print("Mean F1 score:", np.mean(cv_scores))
#
# # ----------------------------------------------------------------------------------
# # 14. Feature Importance Analysis
# # ----------------------------------------------------------------------------------
#
# # Since RandomForestClassifier is the meta-classifier, we can get feature importances
# importances = stacking_clf.final_estimator_.feature_importances_
#
# # Get feature names
# feature_names_text = tfidf_vectorizer.get_feature_names_out()
# feature_names_domain = domain_features
# feature_names = np.concatenate((feature_names_text, feature_names_domain))
#
# # Create a DataFrame for feature importances
# feature_importances = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# })
#
# # Sort features by importance
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
#
# # Display top 20 features
# print("\nTop 20 Important Features:")
# print(feature_importances.head(20))


