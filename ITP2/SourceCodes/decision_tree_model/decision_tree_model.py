import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load SMS and Email data
sms_data = pd.read_csv('/Users/jccasanas/Documents/ITP2/cleaned_sms.csv')
email_data = pd.read_csv('/Users/jccasanas/Documents/ITP2/Phishing_Email.csv')

# Fill missing values
sms_data.fillna('', inplace=True)
email_data.fillna('', inplace=True)

# Combine datasets for a unified approach if feasible
combined_texts = pd.concat([sms_data['TEXT'], email_data['Email Text']])
combined_labels = pd.concat([sms_data['LABEL'], email_data['Email Type']])

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)  # Limit the number of features
features = vectorizer.fit_transform(combined_texts)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(features, combined_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Set up a grid of parameters to test
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Make predictions and evaluate the model
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
