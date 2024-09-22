import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score

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
X = data['TEXT']
y = data['Label']


# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Transform the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_smishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting smishing messages (label=1): {precision_smishing:.4f}")
