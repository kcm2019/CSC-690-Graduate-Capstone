# app.py
from flask import Flask, render_template, request
import time
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Function to load documents 
def load_documents(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cuda:0"}) # "cpu" if no GPU exists
    return embeddings

# Function to create vector store
def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

# Function to create LLMS model
def create_llms_model():
    llm = CTransformers(model="/home/kurtmuller/Documents/Projects/AIJobInterviewPrepBot-master/mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm

# Loading of documents
documents = load_documents("./test.txt")

# Split text into chunks
text_chunks = split_text_into_chunks(documents)

# Create embeddings
embeddings = create_embeddings()

# Create vector store
vector_store = create_vector_store(text_chunks, embeddings)

# Create LLMS model
llm = create_llms_model()

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

# Define chat function
def conversation_chat(query):
    start_time = time.time()  # Start time
    result = chain({"question": query, "chat_history": []})  # Initialize chat history as empty list
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return result["answer"], elapsed_time

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/chat", methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        query = request.form['query']
        response, query_time = conversation_chat(query)
        return render_template('chat.html', query=query, response=response, query_time=query_time)
    else:
        return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
