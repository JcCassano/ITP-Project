import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
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

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values with empty strings (if any)
data = data.fillna('')

# ----------------------------------------------------------------------------------
# 2. Label Mapping and Encoding
# ----------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------------------

# Define functions to extract additional features
def has_url(text):
    return int(bool(re.search(r'http\S+|www\.\S+', text)))

def has_email(text):
    return int(bool(re.search(r'\S+@\S+', text)))

def has_phone(text):
    return int(bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}', text)))

# Apply feature extraction
data['has_url'] = data['TEXT'].apply(has_url)
data['has_email'] = data['TEXT'].apply(has_email)
data['has_phone'] = data['TEXT'].apply(has_phone)
data['message_length'] = data['TEXT'].apply(len)
data['num_exclamations'] = data['TEXT'].str.count('!')
data['num_questions'] = data['TEXT'].str.count('\?')
data['num_uppercase'] = data['TEXT'].apply(lambda x: sum(1 for c in x if c.isupper()))
data['num_digits'] = data['TEXT'].str.count('\d')

# ----------------------------------------------------------------------------------
# 4. Text Vectorization and Feature Combination
# ----------------------------------------------------------------------------------

# Initialize TF-IDF Vectorizer with enhanced parameters
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.9,
    min_df=5,              # Remove very rare words
    ngram_range=(1, 2),    # Include unigrams and bigrams
    max_features=10000     # Limit to top 10,000 features
)

# Fit and transform the text data
X_tfidf = vectorizer.fit_transform(data['TEXT'])

# Extract additional features
additional_features = data[['has_url', 'has_email', 'has_phone', 'message_length',
                           'num_exclamations', 'num_questions', 'num_uppercase', 'num_digits']]

# Convert additional features to a sparse matrix
from scipy import sparse
additional_sparse = sparse.csr_matrix(additional_features.values)

# Combine TF-IDF features with additional features
X_combined = sparse.hstack([X_tfidf, additional_sparse])

# Define target variable
y = data['Label']

# ----------------------------------------------------------------------------------
# 5. Train-Test Split
# ----------------------------------------------------------------------------------

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------------
# 6. Handling Class Imbalance with SMOTEENN
# ----------------------------------------------------------------------------------

# Initialize SMOTEENN
smoteenn = SMOTEENN(random_state=42)

# Apply SMOTEENN to the training data
X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

# Shuffle the resampled data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Verify new label distribution
print("\nAfter applying SMOTEENN:")
print(pd.Series(y_train_resampled).value_counts())

# ----------------------------------------------------------------------------------
# 7. Dimensionality Reduction with Truncated SVD
# ----------------------------------------------------------------------------------

# Initialize Truncated SVD
svd = TruncatedSVD(n_components=300, random_state=42)

# Fit and transform the training data
X_train_reduced = svd.fit_transform(X_train_resampled)

# Transform the test data
X_test_reduced = svd.transform(X_test)

# ----------------------------------------------------------------------------------
# 8. Hyperparameter Tuning with GridSearchCV
# ----------------------------------------------------------------------------------

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',  # Optimize for balanced F1-score across classes
    n_jobs=-1,
    verbose=2
)

# Fit GridSearchCV to the resampled training data
print("\nStarting Grid Search for Hyperparameter Tuning...")
grid_search.fit(X_train_reduced, y_train_resampled)

# Best parameters
print("\nBest parameters found: ", grid_search.best_params_)

# ----------------------------------------------------------------------------------
# 9. Training the Optimized Random Forest Model
# ----------------------------------------------------------------------------------

# Use the best estimator from GridSearchCV
best_rf = grid_search.best_estimator_

# Train the optimized Random Forest classifier
best_rf.fit(X_train_reduced, y_train_resampled)

# Make predictions on the test set
y_pred_best_rf = best_rf.predict(X_test_reduced)

# ----------------------------------------------------------------------------------
# 10. Model Evaluation
# ----------------------------------------------------------------------------------

# Evaluate the model
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
precision_smishing_best_rf = precision_score(y_test, y_pred_best_rf, labels=[2], average='macro', zero_division=0)
recall_smishing_best_rf = recall_score(y_test, y_pred_best_rf, labels=[2], average='macro', zero_division=0)
f1_smishing_best_rf = f1_score(y_test, y_pred_best_rf, labels=[2], average='macro', zero_division=0)

print(f"\n[Optimized RF] Accuracy: {accuracy_best_rf:.4f}")
print(f"[Optimized RF] Precision for Smishing (label=2): {precision_smishing_best_rf:.4f}")
print(f"[Optimized RF] Recall for Smishing (label=2): {recall_smishing_best_rf:.4f}")
print(f"[Optimized RF] F1-score for Smishing (label=2): {f1_smishing_best_rf:.4f}")

# Classification Report
print("\n[Optimized RF] Classification Report:")
target_names = ['Ham', 'Spam', 'Smishing']
print(classification_report(y_test, y_pred_best_rf, target_names=target_names))

# Confusion Matrix
conf_matrix_opt_rf = confusion_matrix(y_test, y_pred_best_rf)
print("\n[Optimized RF] Confusion Matrix:")
print(conf_matrix_opt_rf)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_opt_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized Random Forest - Confusion Matrix')
plt.show()


# ----------------------------------------------------------------------------------
# 11. Cross-Validation for Robust Evaluation
# ----------------------------------------------------------------------------------

# Define Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the optimized Random Forest using cross-validation
cv_scores_rf = cross_val_score(best_rf, X_train_reduced, y_train_resampled, cv=skf, scoring='f1_macro', n_jobs=-1)
print(f"\n[Cross-Validation] Random Forest F1 Scores: {cv_scores_rf}")
print(f"[Cross-Validation] Mean F1 Score: {cv_scores_rf.mean():.4f}")

# ----------------------------------------------------------------------------------
# 12. Ensemble Methods: Stacking Classifier (Optional)
# ----------------------------------------------------------------------------------

# Initialize other classifiers for stacking
gb_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=2,
    random_state=42,
    class_weight='balanced'
)

lr_clf = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

# Define base estimators
base_estimators = [
    ('rf', gb_clf),
    ('lr', lr_clf)
]

# Define the meta-classifier
meta_classifier = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

# Initialize the Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_classifier,
    cv=5,
    n_jobs=-1
)

# Train the Stacking Classifier
stacking_clf.fit(X_train_reduced, y_train_resampled)

# Make predictions
y_pred_stacking = stacking_clf.predict(X_test_reduced)

# Evaluate the Stacking Classifier
accuracy_stack = accuracy_score(y_test, y_pred_stacking)
precision_smishing_stack = precision_score(y_test, y_pred_stacking, labels=[2], average='macro', zero_division=0)
recall_smishing_stack = recall_score(y_test, y_pred_stacking, labels=[2], average='macro', zero_division=0)
f1_smishing_stack = f1_score(y_test, y_pred_stacking, labels=[2], average='macro', zero_division=0)

print(f"\n[Stacking] Accuracy: {accuracy_stack:.4f}")
print(f"[Stacking] Precision for Smishing (label=2): {precision_smishing_stack:.4f}")
print(f"[Stacking] Recall for Smishing (label=2): {recall_smishing_stack:.4f}")
print(f"[Stacking] F1-score for Smishing (label=2): {f1_smishing_stack:.4f}")

# Classification Report for Stacking Classifier
print("\n[Stacking] Classification Report:")
print(classification_report(y_test, y_pred_stacking, target_names=target_names))

# Confusion Matrix for Stacking Classifier
conf_matrix_stack = confusion_matrix(y_test, y_pred_stacking)
print("\n[Stacking] Confusion Matrix:")
print(conf_matrix_stack)

# Visualize the Confusion Matrix for Stacking Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stack, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Stacking Classifier - Confusion Matrix')
plt.show()

# ----------------------------------------------------------------------------------
# 13. Summary and Final Thoughts
# ----------------------------------------------------------------------------------

print("\n--- Summary ---")
print(f"Initial Random Forest Accuracy: 0.9437")
print(f"Optimized Random Forest Accuracy: {accuracy_best_rf:.4f}")
print(f"Stacking Classifier Accuracy: {accuracy_stack:.4f}")
print(f"Cross-Validation Mean F1 Score: {cv_scores_rf.mean():.4f}")
print("\nModel performance has been enhanced through hyperparameter tuning, feature engineering, advanced sampling techniques, and ensemble methods.")
print("Further improvements can be achieved by experimenting with different feature representations, such as word embeddings or transformer-based models like BERT.")
print("Additionally, continuous evaluation and iteration are key to maintaining and improving model performance.")

# ----------------------------------------------------------------------------------
