import spacy
from tqdm import tqdm

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Function to count tokens in a document
def count_tokens(document):
    # Process the text with spaCy
    doc = nlp(document)
    
    # Get the number of tokens
    token_count = len(doc)
    
    return token_count

# Read the input from a text file
file_path = "test.txt"  # Replace with your file path
with open(file_path, "r", encoding="utf-8") as file:
    documents = file.readlines()

# Count tokens for each document with progress bar
total_token_count = 0
with tqdm(total=len(documents), desc="Counting Tokens") as pbar:
    for doc in documents:
        total_token_count += count_tokens(doc)
        pbar.update(1)

print("Total token count for all documents:", total_token_count)
