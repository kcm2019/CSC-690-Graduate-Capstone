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
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    "body"
                )
            ),
        )
        return loader.load()
    except Exception as e:
        print(f"Error downloading content from {url}: {e}")
        return None


# Example usage
url_list = read_urls_from_file("texturls.txt")

# Total number of URLs
total_urls = len(url_list)

docs = []
# Downloading & extracting info from the site
with tqdm(total=total_urls) as pbar:
    for url in url_list:
        loaded_content = process_url(url)
        if loaded_content:
            docs += loaded_content
        pbar.update()

# Assuming you have docs loaded correctly
# Use pickle.dumps with a protocol version for backwards compatibility
with open("adelphitestdata.pickle", "wb") as f:
    pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)

# Later, to read:
with open("adelphitestdata.pickle", "rb") as f:
    loaded_docs = pickle.load(f)

print(type(loaded_docs))


""" 
# Convert nested objects to JSON-compatible types
#json_data = json.dumps(docs, default=lambda o: str(o))
json_data = json.dumps(docs, default=lambda o: o.__dict__)


# Write the JSON string to a file
with open("data.json", "w") as f:
    f.write(json_data)

# Later, to read the data:
with open("data.json", "r") as f:
    json_string = f.read()
    loaded_data = json.loads(json_string)
    #data_list = ast.literal_eval(loaded_data)

print("loaded data: " + str(type(loaded_data)))
#print("ast; " + type(data_list)) 


print(json.dumps(loaded_data, indent=4))

 with open("data.json", "r") as f:
    json_string = f.read()
    loaded_data = json.loads(json_string)
    data_list = ast.literal_eval(loaded_data) 


splits = text_splitter.split_documents(loaded_docs)

print(str(type(splits)) + " \n ***END SPLITS***")
# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Use the RAG chain
result = rag_chain("Tell me something about Adelphi")
print(result)  """