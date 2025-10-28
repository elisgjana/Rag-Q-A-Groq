import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
import time
from dotenv import load_dotenv

# ---------------------------
# ✅ Load environment variables
# ---------------------------
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------------------
# Initialize session state keys early
# (prevents AttributeError if accessed before creation)
# ---------------------------
if "vectors" not in st.session_state:
    st.session_state.vectors = None  # <-- ensures existence even before building

# ---------------------------
# Setup LLM and prompt
# ---------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# ---------------------------
# Function to build the vector DB (called once)
# ---------------------------
def create_vector_embedding():
    # Always (re)build the vector DB when called
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data Ingestion step
    st.session_state.docs = st.session_state.loader.load()  # Document Loading
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )
    st.success("✅ Vector Database is ready!")

# ---------------------------
# Streamlit UI
# ---------------------------
st.image("icon.png", width=100)
st.title("Research Assistant")
st.subheader("RAG Document Q&A With Groq LLM And OpenAI Embedding - by Elis")

user_prompt = st.text_input("Enter your query from the research papers")

# ---------------------------
# Build the vector DB on button click
# ---------------------------
if st.button("Document Embedding"):
    with st.spinner("Building the vector database..."):
        create_vector_embedding()

# ---------------------------
# Handle user query
# ---------------------------
if user_prompt:
    # Guard clause: ensure vectors exist before use
    if st.session_state.vectors is None:
        st.warning("Please click **Document Embedding** first to build the vector database.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write(f"⏱️ Response time: {time.process_time() - start:.2f}s")

        # Display the answer
        st.subheader("Answer:")
        st.write(response["answer"])

        # Display similar documents
        with st.expander("📄 Document similarity search results"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---")
else:
    st.info("Please enter a question above to query the research papers.")
