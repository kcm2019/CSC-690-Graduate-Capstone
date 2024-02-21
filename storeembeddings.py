import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
import pickle

start_time = time.time()

""" # Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="mistral")

with open("embeddings.pickle", "wb") as f:
    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

# Later, to read:
with open("adelphidata.pickle", "rb") as f:
    loaded_embedding = pickle.load(f)

print(type(loaded_embedding)) """

with open("adelphidata.pickle", "rb") as f:
    loaded_docs = pickle.load(f)

with open("embeddings.pickle", "rb") as f:
    loaded_embeddings = pickle.load(f)

#embeddings = OllamaEmbeddings(model="mistral")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(loaded_docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=loaded_embeddings)

with open("vectorstore.pickle", "wr") as f:
    pickle.dump(vectorstore, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("vectorstore.pickle", "rb") as f:
    loaded_vectorstore = pickle.load(f)

print(type(loaded_vectorstore))

end__time = time.time()
print(end__time-start_time)