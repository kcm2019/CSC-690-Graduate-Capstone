# AdelphiLM: An AI Chatbot for Adelphi University

This project aims to develop AdelphiLM, an advanced chatbot for Adelphi University that can answer user queries directly, drawing information from the university's website.

## Why AdelphiLM?

Currently, Adelphi's existing chatbot, Adele, simply provides links and expects users to find the answers themselves. AdelphiLM aspires to be a more comprehensive and user-friendly experience, similar to ChatGPT, by:

- **Understanding human language:** AdelphiLM will be trained to interpret user queries in natural language.
- **Accessing and processing Adelphi website data:** AdelphiLM will be trained on a dataset of relevant content from the Adelphi University website, enabling it to answer questions directly.
- **Providing informative responses:** AdelphiLM will strive to deliver informative and helpful responses based on its understanding of the user's intent.

## Project Goals

The primary goals of this project are:

- To create a user-friendly chatbot interface for Adelphi University.
- To empower AdelphiLM to answer a wide range of questions related to the university using information from its official website.
- To leverage the capabilities of PyTorch for deep learning and Flask for the web interface.

## Technical Stack

This project utilizes the following technologies:

- **PyTorch:** A deep learning framework for training the language model.
- **Mistral 7B:** A pre-trained factual language model (optional, depending on implementation choices).
- **RAG (Retriever-Reader Generative Transformer):** A model architecture for leveraging retrieval and generation techniques.
- **Flask:** A lightweight web framework for creating the chatbot's user interface.

# Program Set Up

## Packages To Install

The following Python packages must be installed to successfully run the program:

- `langChain`
- `faiss-gpu`
- `transformers`
- `torch` (with CUDA support)
- `sentence-transformers`
- `ctransformers`
- `llama-cpp-python`
- `pandas`
- `huggingfacecli`

## Additional Preparation

### Huggingface CLI

You must set up `huggingfacecli` and set up your account within the command line tool. This will enable us to use the `sentence-transformers` library to use the `all-MiniLM-L6-v2` sentence transformer that enables the creation of embeddings for the vectorstore. This is essential to be able to convert our scrapped information from Adelphi’s website and convert it into a vector store that the RAG Chain can pull context from.

### Mistral 7B Download

The next main component of the program is the LLM itself. The one chosen is Mistral 7B, specifically `mistral-7b-instruct-v0.1.Q4_K_M.gguf`. This version of Mistral 7B was the perfect mix of being a medium-weight model that could run on the available hardware, while being the perfect start for a RAG Chain implementation. Download and store this in a file. Change the path found in `app.py` (line 53), linked here, to be able to point to the correct location in your files.

## How to Run

1. First, download the project from GitHub.
2. Once you have installed the necessary components and completed the setup steps above, you can run the `app.py` file. Running this file will set up the LLM, RAG Chain, and Flask application.
3. After a few minutes of setup, the website will be hosted locally on your machine and accessible from the link that Flask provides in the terminal.
4. From here, you will see the Adelphi University website homepage that has been modified to replace Adele with AdelphiLM in the lower right-hand corner.

