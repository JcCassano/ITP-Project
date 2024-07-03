from flask import Flask, render_template, request, redirect, url_for, flash
import imaplib
import email
from email.header import decode_header
import csv
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("cybersectony/phishing-email-detection-distilbert")

def clean_html(html_content):
    cleaned_text = re.sub(r'<[^>]*>', '', html_content)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def save_emails_to_csv(email_user, email_pass, num_emails=5):
    # Connect to the Outlook email server
    mail = imaplib.IMAP4_SSL("outlook.office365.com")
    mail.login(email_user, email_pass)
    mail.select("inbox")

    # Search for all emails in the inbox
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()[:num_emails]

    with open('emails.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['body']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        cleaned_body = clean_html(body)
                        writer.writerow({'body': cleaned_body})
            else:
                body = msg.get_payload(decode=True).decode()
                cleaned_body = clean_html(body)
                writer.writerow({'body': cleaned_body})

    mail.logout()

def classify_email_chunk(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(-1)
    return predictions.item()

def classify_email(text):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+512] for i in range(0, len(tokens), 512)]
    chunk_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    phishing_predictions = sum(classify_email_chunk(chunk) for chunk in chunk_texts)
    return 1 if phishing_predictions > 0 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_emails', methods=['GET', 'POST'])
def extract_emails():
    if request.method == 'POST':
        email_user = request.form.get('email_user')
        email_pass = request.form.get('email_pass')
        num_emails = int(request.form.get('num_emails', 10))
        save_emails_to_csv(email_user, email_pass, num_emails)
        flash('Emails extracted successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('extract.html')

@app.route('/classify_emails', methods=['GET', 'POST'])
def classify_emails():
    if request.method == 'POST':
        df = pd.read_csv("emails.csv")
        df['prediction'] = df['body'].apply(classify_email)
        df.to_csv("emails_with_predictions.csv", index=False)
        flash('Emails classified successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('classify.html')

if __name__ == '__main__':
    app.run(debug=True)
