import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load dataset
data = pd.read_csv('../../Dataset/cleaned_sms.csv')
X = data['TEXT']
y = data['LABEL'].apply(lambda x: 1 if x == "Smishing" else 0)  # 1 for Smishing, 0 for Ham

# Fill any NaN values in the text column
X = X.fillna('')

# Feature Extraction
vectorizer = CountVectorizer(max_features=1000)
X_vect = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Machine Learning Models
# 1. Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 3. Support Vector Machine (SVM)
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Ensemble Model: Voting Classifier
voting_clf = VotingClassifier(estimators=[('nb', nb_model), ('rf', rf_model), ('svm', svm_model)], voting='soft')
voting_clf.fit(X_train, y_train)
ensemble_pred = voting_clf.predict(X_test)

# Evaluate Machine Learning Models
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_pred))
print("\nClassification Report:\n", classification_report(y_test, ensemble_pred))

# Deep Learning Model: Neural Network as per paper recommendations
# Creating a simple NN model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train.toarray(), y_train, epochs=10, batch_size=100, validation_data=(X_test.toarray(), y_test))

# Neural Network Evaluation
nn_loss, nn_accuracy = nn_model.evaluate(X_test.toarray(), y_test)
print("\nNeural Network Accuracy:", nn_accuracy)
