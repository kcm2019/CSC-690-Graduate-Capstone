import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm  # Import tqdm for progress bar
import pickle
import ollama
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

""" YOUTUBE LINK: https://www.youtube.com/watch?v=4HfSfFvLn9Q
    PIP INSTALL: pip3 install langchain beautifulsoup4 chromadb gradio tqdm
    Project Link: https://mer.vin/2024/01/ollama-rag/
"""

def read_urls_from_file(filename):
  """
  Reads URLs from a text file and returns them as a list.

  Args:
    filename: The name of the text file containing URLs.

  Returns:
    A list of URLs from the text file.
  """
  with open(filename, "r") as f:
    return [line.strip() for line in f.readlines()]

def process_url(url):
  loader = WebBaseLoader(
      web_paths=(url,),
      bs_kwargs=dict(
          parse_only=bs4.SoupStrainer(
              'body'
          )
      ),
  )
  return loader.load()

# Example usage
url_list = read_urls_from_file("texturls.txt")

# Total number of URLs
total_urls = len(url_list)

docs=[]
# Downloading & extracting info from the site
with tqdm(total=total_urls) as pbar:
  for url in url_list:
    loaded_content = process_url(url)
    docs += loaded_content
    pbar.update()  # Update progress bar

print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# need to capture & store: retriever
  
# Create Ollama embeddings and vector store
print("Creating embedding and vectorstore...")
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Create the retriever
print("Create retriever...")
retriever = vectorstore.as_retriever()

#Store the retriever
print("Storing retriever...")
with open("adelphiretriever.pickle", "wb") as f:
  pickle.dump(retriever, f, protocol=pickle.HIGHEST_PROTOCOL)

# Later, to read:
print("Reading retriever...")
with open("adelphiretriever.pickle", "rb") as f:
  loaded_retriever = pickle.load(f)

#Format the docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = loaded_retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Use the RAG chain
result = rag_chain("Tell me something about Adelphi")
print(result)

""" # Assuming you have docs loaded correctly
# Use pickle.dumps with a protocol version for backwards compatibility
with open("adelphidata.pickle", "wb") as f:
    pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)

# Later, to read:
with open("adelphidata.pickle", "rb") as f:
    loaded_docs = pickle.load(f) """
