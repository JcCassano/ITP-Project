import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.decomposition import TruncatedSVD

# Load the dataset
data = pd.read_csv('../../Dataset/Phishing_Email.csv')

# Fill missing values with empty strings
data = data.fillna('')

# Map labels: 'Phishing Email' to 1, 'Safe Email' to 0
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
data['Email Type'] = data['Email Type'].map(label_mapping)

# Verify label encoding
print("\nLabel distribution:")
print(data['Email Type'].value_counts())

# Define features and target variable
X = data['Email Text']
y = data['Email Type']

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Transform the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Reduce dimensionality with TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_reduced = svd.transform(X_test_tfidf)

# Initialize the KNN classifier with cosine distance
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')

# Train the classifier
knn.fit(X_train_reduced, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_reduced)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision_phishing = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy of the model: {accuracy:.4f}")
print(f"Precision for detecting phishing emails (label=1): {precision_phishing:.4f}")
