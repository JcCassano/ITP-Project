import pandas as pd
import re


# Function to clean the text column (Column B)
def clean_text(text):
    if isinstance(text, str):
        # Replace encoded HTML entities like &lt; and &gt;
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)

        # Remove unwanted characters like ï¿½, Â, and others
        text = re.sub(r'[Âï¿½]+', '', text)

        # Remove any non-ASCII characters that might cause issues
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Replace multiple spaces or newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load the CSV file
file_path = '../../Dataset/sms.csv'
df = pd.read_csv(file_path)

# Apply the cleaning function to the 'TEXT' column (Column B)
df['TEXT'] = df['TEXT'].apply(clean_text)

# Save the cleaned data to a new CSV file
output_file_path = '../../Dataset/cleaned_sms.csv'
df.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")
