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
from flask import jsonify

# Function to print execution timeh
def print_execution_time(message, start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{message}: {elapsed_time} seconds")


# Function to load documents 
def load_documents(filepath):
    start_time = time.time()
    loader = TextLoader(filepath)
    documents = loader.load()
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
    llm = CTransformers(model="/home/kurtmuller/Documents/Projects/AIJobInterviewPrepBot-master/mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 512, 'temperature': 0.01})
    print_execution_time("\nCreated LLM", start_time)
    return llm

# Loading of documents
# adelphidata.txt
documents = load_documents("./adelphidata.txt")

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

cont = True
while cont:
    user_input = input("Type 'exit' to quit: ")
    print(conversation_chat(user_input))
    if user_input.lower() == 'exit':
        cont = False

#Simple Implementation of Testing
""" from sklearn.metrics import accuracy_score

# Define a list of test queries and their expected responses
test_queries = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the square root of 16?"]
expected_responses = ["Paris", "Harper Lee", "4"]

# Use the conversation_chat function to generate responses for the test queries
generated_responses = [conversation_chat(query)[0] for query in test_queries]

# Compare the generated responses with the expected responses
accuracy = accuracy_score(expected_responses, generated_responses)

print(f"Accuracy of the ConversationalRetrievalChain: {accuracy * 100}%") """

# BLEU testing
"""
Sure, you can use the BLEU (Bilingual Evaluation Understudy) score to evaluate the quality of the generated responses. The BLEU score is a metric used in machine translation to measure the quality of translations by comparing them to a set of reference translations. Here’s how you might do this:
Python

from nltk.translate.bleu_score import sentence_bleu

# Define a list of test queries and their expected responses
test_queries = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the square root of 16?"]
expected_responses = [["Paris"], ["Harper Lee"], ["4"]]

# Use the conversation_chat function to generate responses for the test queries
generated_responses = [conversation_chat(query)[0] for query in test_queries]

# Calculate the BLEU scores for the generated responses
bleu_scores = [sentence_bleu([expected], generated) for expected, generated in zip(expected_responses, generated_responses)]

# Print the average BLEU score
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score: {average_bleu_score}")

In this code, sentence_bleu is used to calculate the BLEU score for each generated response compared to the expected response. The BLEU score ranges from 0 to 1, where 1 means a perfect match with the reference translation. Note that the expected responses are lists of words, as the BLEU score considers individual words and their order.
Please note that while BLEU is a popular metric, it has its limitations and might not always reflect the quality of a conversational AI perfectly. It’s always a good idea to complement it with other evaluation methods. Also, keep in mind that the BLEU score is more meaningful when calculated on a large number of samples. The three samples used here are just for demonstration purposes. You should use a larger and more diverse set of test queries for a more reliable evaluation.
"""

#BLEURT: https://towardsdatascience.com/how-to-measure-the-success-of-your-rag-based-llm-system-874a232b27eb