"""
Document Q&A Assistant with Local Vector Storage
Production-ready application for document-based Q&A using Amazon Bedrock and FAISS
"""

import os
from typing import Optional

import boto3
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Load environment variables
load_dotenv()

# Configuration
PAGE_TITLE = os.getenv("PAGE_TITLE", "Document Q&A Assistant")
APP_TITLE = os.getenv("APP_TITLE", "Document Q&A Assistant")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-instant-v1")
DEFAULT_PDF_PATH = os.getenv("DEFAULT_PDF_PATH", "documents/sample.pdf")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "1.0"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Model parameters
MODEL_KWARGS = {
    "max_tokens_to_sample": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P
}


@st.cache_resource
def get_bedrock_client():
    """Initialize and return Bedrock client."""
    return boto3.client('bedrock-runtime')


@st.cache_resource
def get_llm():
    """Initialize and return Bedrock LLM."""
    client = get_bedrock_client()
    llm = Bedrock(model_id=DEFAULT_MODEL_ID, client=client)
    llm.model_kwargs = MODEL_KWARGS
    return llm


@st.cache_resource
def get_embeddings():
    """Initialize and return Bedrock embeddings."""
    client = get_bedrock_client()
    return BedrockEmbeddings(client=client)


@st.cache_resource
def load_vector_store(pdf_path: str):
    """Load PDF and create FAISS vector store."""
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        st.stop()
    
    embeddings = get_embeddings()
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    vectorstore = FAISS.from_documents(pages, embeddings)
    return vectorstore


def get_prompt_template():
    """Create and return the prompt template."""
    template = """
Human:
    You are a conversational assistant designed to help answer questions from documents.
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like "according to the training or handbook or based on or according to the information provided...".

    <Information>
    {info}
    </Information>

    {input}

Assistant:
"""
    return PromptTemplate(
        input_variables=['info', 'input'],
        template=template
    )


def main():
    """Main application entry point."""
    # Configure Streamlit app
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📖",
        layout="wide",
    )
    st.title(APP_TITLE)

    # Display configuration info in sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Model: {DEFAULT_MODEL_ID}")
        st.info(f"Max Tokens: {MAX_TOKENS}")
        st.info(f"Temperature: {TEMPERATURE}")
        st.info(f"Top P: {TOP_P}")
        st.info(f"Top K Results: {TOP_K_RESULTS}")
        
        st.header("Document Upload")
        pdf_file = st.file_uploader("Upload PDF document", type=['pdf'])
        
        if pdf_file:
            # Save uploaded file
            pdf_path = f"temp_{pdf_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            st.success(f"Uploaded: {pdf_file.name}")
        else:
            pdf_path = DEFAULT_PDF_PATH
            st.info(f"Using default: {DEFAULT_PDF_PATH}")

    # Initialize components
    try:
        llm = get_llm()
        vectorstore = load_vector_store(pdf_path)
        prompt_template = get_prompt_template()
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.stop()

    # Create LLM chain
    question_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="answer"
    )

    # Set up message history
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you with the document?")

    # Render current messages
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Handle user input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.chat_message("human").write(prompt)

        try:
            # Perform similarity search
            with st.spinner("Searching document..."):
                docs = vectorstore.similarity_search_with_score(prompt, k=TOP_K_RESULTS)
                info = ""
                for doc in docs:
                    info += doc[0].page_content + '\n'

            # Invoke LLM
            with st.spinner("Generating answer..."):
                output = question_chain.invoke({"input": prompt, "info": info})

            # Add to history
            msgs.add_user_message(prompt)
            msgs.add_ai_message(output['answer'])

            # Display the output
            st.chat_message("ai").write(output['answer'])

            # Optionally display source documents
            if st.checkbox("Show source documents"):
                with st.expander("Source Documents"):
                    for i, (doc, score) in enumerate(docs, 1):
                        st.write(f"**Source {i}** (Score: {score:.4f}):")
                        st.write(doc.page_content)
                        st.write("---")

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    main()
