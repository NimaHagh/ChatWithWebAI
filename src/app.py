import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables, necessary for API keys and configuration
load_dotenv()

def get_vectorstore_from_url(url):
    """
    Retrieves and processes text from a given URL to create a searchable vector store.
    
    Parameters:
    url (str): The URL from which to fetch the text.

    Returns:
    Chroma: A vector store built from the processed text documents, or None if an error occurs.
    """
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
        return vector_store
    except Exception as e:
        st.error(f"Error loading content from URL: {e}")
        return None

def get_context_retriever_chain(vector_store):
    """
    Configures a retrieval chain with a language model for processing user inputs.
    
    Parameters:
    vector_store (Chroma): The vector store used for information retrieval.

    Returns:
    The configured retriever chain, or None if an error occurs.
    """
    if not vector_store:
        return None

    try:
        llm = ChatOpenAI()
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        return create_history_aware_retriever(llm, retriever, prompt)
    except Exception as e:
        st.error(f"Error setting up the retriever chain: {e}")
        return None

def get_conversational_rag_chain(retriever_chain):
    """
    Configures a retrieval-augmented generation chain for answering user questions.
    
    Parameters:
    retriever_chain: The retriever chain to use for fetching context.

    Returns:
    A retrieval chain designed for generating conversational responses.
    """
    try:
        llm = ChatOpenAI()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    except Exception as e:
        st.error(f"Error setting up the conversational chain: {e}")
        return None

def get_response(user_input):
    """
    Processes the user's input message and generates a response.
    
    Parameters:
    user_input (str): The user's input message.

    Returns:
    str: The generated response from the AI, or a default message if an error occurs.
    """
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.error("Vector store is not initialized.")
        return "No response generated."

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    if retriever_chain is None:
        return "No response generated."

    try:
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        return response['answer']
    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "No response generated."

# Streamlit app configuration
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Main app logic
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
