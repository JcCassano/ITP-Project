from flask import Flask, render_template, request, redirect, url_for, flash
import imaplib
import email
from email.header import decode_header
import csv
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import os
import subprocess
import win32com.client
import pythoncom

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def clean_html(html_content):
    cleaned_text = re.sub(r'<[^>]*>', '', html_content)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def save_emails_to_csv(num_emails=5, read_all=False):
    pythoncom.CoInitialize()

    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)  # "6" refers to the inbox
    messages = inbox.Items
    messages.Sort("[ReceivedTime]", True)  # Sort by received time, descending

    email_count = messages.Count if read_all else min(num_emails, messages.Count)

    with open('emails.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['body']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i in range(email_count):
            message = messages[i]
            cleaned_body = clean_html(message.Body)
            writer.writerow({
                'body': cleaned_body
            })

    print(f"Saved {email_count} emails to emails.csv")

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
        num_emails = int(request.form.get('num_emails', 10))
        save_emails_to_csv(num_emails, read_all=False)
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

# Integrate scripts
@app.route('/run_crawler')
def run_crawler():
    result = subprocess.run(['python', 'scripts/crawler.py'], capture_output=True, text=True)
    flash(result.stdout, 'success')
    return redirect(url_for('index'))

@app.route('/run_cvedbscript')
def run_cvedbscript():
    result = subprocess.run(['python', 'scripts/CVEDBScript.py'], capture_output=True, text=True)
    flash(result.stdout, 'success')
    return redirect(url_for('index'))

@app.route('/run_policyquery')
def run_policyquery():
    result = subprocess.run(['python', 'scripts/policyquerygpt4.py'], capture_output=True, text=True)
    flash(result.stdout, 'success')
    return redirect(url_for('index'))

@app.route('/run_outlook')
def run_outlook():
    result = subprocess.run(['python', 'scripts/outlook.py'], capture_output=True, text=True)
    flash(result.stdout, 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
