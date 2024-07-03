import win32com.client
import csv
import re

def clean_html(html_content):
    # Clean HTML tags and replace newlines with spaces
    cleaned_text = re.sub(r'<[^>]*>', '', html_content)  # Remove HTML tags
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
    return cleaned_text.strip()  # Trim leading and trailing spaces

def save_emails_to_csv(num_emails=5, read_all=False):
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
            message = messages[i + 1]  # Outlook is 1-based index
            cleaned_body = clean_html(message.Body)
            writer.writerow({
                'body': cleaned_body
            })

    print(f"Saved {email_count} emails to emails.csv")

# Call the function with desired parameters
save_emails_to_csv(num_emails=10, read_all=False)
