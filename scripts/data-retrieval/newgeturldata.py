import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import pickle
from langchain_core.documents import Document

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from HTML and preprocess
            text = soup.get_text(separator=" ")
            text = preprocess_text(text)
            return text
    except Exception as e:
        print(f"Error fetching URL: {url}. {e}")
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
    documents = []

    with open(input_file, 'r') as f:
        urls = f.read().splitlines()

    for url in tqdm(urls, desc="Processing URLs"):
        text = fetch_text_from_url(url)
        if text:
            # Create a Document object for each URL with metadata
            document = Document(page_content=text, text=text, metadata={"url": url})
            documents.append(document)

    # Save the list of Document objects
    with open(output_file, 'wb') as f:
        # Pickle the list of Document objects
        pickle.dump(documents, f)

if __name__ == "__main__":
    input_file = "adelphiurls.txt"
    output_file = "output.pkl"
    main(input_file, output_file)
