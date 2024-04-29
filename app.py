import pickle
from flask import Flask, render_template, request, jsonify
import time
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

app = Flask(__name__)

# Function to print execution time
def print_execution_time(message, start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{message}: {elapsed_time} seconds")

# Function to load documents from pickle object
def load_documents_from_pickle(pickle_file):
    start_time = time.time()
    with open(pickle_file, 'rb') as f:
        documents = pickle.load(f)
    print_execution_time("\nLoaded Documents", start_time)
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    print_execution_time("\nSplit Text", start_time)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cuda:0"}) # "cpu" if no GPU exists
    print_execution_time("\nCreated Embeddings", start_time)
    return embeddings

# Function to create vector store
def create_vector_store(text_chunks, embeddings):
    start_time = time.time()
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    print_execution_time("\nCreated vector store", start_time)
    return vector_store

# Function to create LLMS model
def create_llms_model():
    start_time = time.time()
    llm = CTransformers(model="/home/kurtmuller/Documents/Projects/AIJobInterviewPrepBot-master/mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 1000, 'temperature': 0.01, 'context_length': 2048})
    print_execution_time("\nCreated LLM", start_time)
    return llm

# Loading of documents from pickle
# "./output.pkl"
documents = load_documents_from_pickle("./output.pkl")

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
    query_with_instructions = query + " Be concise. Stick to the context. If a question that is asked is not found in the context, just say \"I could not find that in the given context.\" Be helpful. Avoid speculation. Use natural language. Provide accurate information. Use a maximum of 5 sentences"
    # Call the chain with the modified query
    result = chain({"question": query_with_instructions, "chat_history": []})  # Initialize chat history as empty list
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return result["answer"], elapsed_time

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        
        # Call the conversation_chat function to get the response and query time
        response, query_time = conversation_chat(query)
        
        # Return the response, query, and query time as JSON data
        return jsonify({'query': query, 'response': response, 'query_time': query_time})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
