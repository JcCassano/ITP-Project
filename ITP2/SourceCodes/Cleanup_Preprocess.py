import pandas as pd
import re
from dateutil import parser

# Function to extract only the email addresses from the 'sender' and 'receiver' columns
def extract_email(text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', str(text))
    if email:
        return email[0]
    return None

# Function to clean up the date to only contain day/month/year
def clean_date(text):
    try:
        parsed_date = parser.parse(text)
        return parsed_date.strftime('%d/%m/%Y')  # Format as day/month/year
    except (parser.ParserError, TypeError):
        return None

# Function to check if subject contains gibberish and remove it
def clean_subject(text):
    # Example heuristic: if the subject contains a high proportion of non-ASCII characters, we consider it gibberish
    if text and isinstance(text, str):
        non_ascii_ratio = sum(1 for char in text if ord(char) > 127) / len(text)
        if non_ascii_ratio > 0.5:  # Tune this threshold as needed
            return None
    return text

# Function to clean body, removing illegible characters while keeping legitimate content
def clean_body(text):
    if text and isinstance(text, str):
        # Remove sequences of non-ASCII characters that are likely gibberish
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Optionally remove long sequences of spaces, tabs, or other control characters
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
        return cleaned_text
    return None


# Load the Excel file
file_path = '../Dataset/Compiled_email.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Apply the cleaning functions
df['sender'] = df['sender'].apply(extract_email)
df['receiver'] = df['receiver'].apply(extract_email)
df['date'] = df['date'].apply(clean_date)
df['subject'] = df['subject'].apply(clean_subject)
df['body'] = df['body'].apply(clean_body)

# Save the cleaned data back to the Dataset folder
output_file_path = '../Dataset/Cleaned_Compiled_email.xlsx'  # Output file path
df.to_excel(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")
