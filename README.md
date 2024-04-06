# Chat with the content of a Website

## Overview
This application leverages Streamlit and various langchain components to enable users to chat with the content of any website. It extracts and processes the text from a given URL to generate responses to user queries, mimicking a conversational agent.
## Features
- Website Text Extraction: Utilizes WebBaseLoader to fetch and process text from the specified website URL.
- Information Retrieval: Builds a vector store using Chroma and OpenAIEmbeddings for efficient text searching.
- Conversational Interface: Implements a retrieval-augmented generation chain to provide contextually relevant answers.
## Diagram
Here is the flow of the application:
![RAG-1 (1)](https://github.com/NimaHagh/ChatWithWebAI/assets/105126750/8fe1e772-baba-4ec9-b413-95ed85f1ec73)

## Installation
1. Clone the repository:
```
git clone [repository-url]
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Start the Streamlit app:
```
streamlit run app.py
```

## Development Setup
- Environment Variables: Store your API keys and configurations in a .env file.
- Vector Store Initialization: The vector store is created by processing the website content and used for information retrieval.
- Retrieval Chain Configuration: Configures the retriever chain with a language model to process user inputs.
