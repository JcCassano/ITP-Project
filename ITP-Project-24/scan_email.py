import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("dima806/phishing-email-detection")
# model = AutoModelForSequenceClassification.from_pretrained("dima806/phishing-email-detection")

tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("cybersectony/phishing-email-detection-distilbert")

# Function to classify an email chunk
def classify_email_chunk(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(-1)
    return predictions.item()

# Function to split email text into chunks and classify each chunk
def classify_email(text):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+512] for i in range(0, len(tokens), 512)]
    chunk_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    phishing_predictions = sum(classify_email_chunk(chunk) for chunk in chunk_texts)
    return 1 if phishing_predictions > 0 else 0

# Load exported emails (assuming a CSV file)
df = pd.read_csv("emails.csv")

# Iterate through each email and classify
for index, row in df.iterrows():
    email_text = row["body"]  # column containing email content
    prediction = classify_email(email_text)

    if prediction == 1:
        print(f"Potential phishing email detected: {email_text}")
    else:
        print(f"Normal email: {email_text}")

# Save the predictions to a new CSV file
df['prediction'] = df['body'].apply(classify_email)
df.to_csv("emails_with_predictions.csv", index=False)
