import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

def fetch_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from HTML and preprocess
        text = soup.get_text(separator=" ")
        text = preprocess_text(text)
        return text
    else:
        return None

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization (split by space)
    tokens = text.split()
    # Join tokens back to text
    text = ' '.join(tokens)
    return text

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        urls = f.read().splitlines()

    with open(output_file, 'w') as f:
        for url in tqdm(urls, desc="Processing URLs"):
            text = fetch_text_from_url(url)
            if text:
                f.write(text + '\n')

if __name__ == "__main__":
    input_file = "adelphiurls.txt"
    output_file = "output.txt"
    main(input_file, output_file)
