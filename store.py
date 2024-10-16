from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma 
import os
import shutil

# Directory to your pdf files
DATA_PATH = "data/"

def load_documents():
    """
    Loads user input and PDF files in a specified directory.
    Returns:
        list[Document]: A list of Document objects containing both user input and PDF documents.
    """
    # Each string will be treated as a separate document
    user_input = [
        "Llamas are members of the camelid family meaning they're pretty closely related to camels.",
        "Cook landed on the moon in the year 1778 after taking a wrong turn to Hawaii. It was hailed as one of history's greatest navigational disasters.",
        ### ADD YOUR OWN TEXT HERE ###
    ]
    documents = []
    for i, text in enumerate(user_input):
        documents.append(Document(page_content=text, metadata={"source": f"User input {i}"}))

    # Initialize PDF loader with specified directory
    document_loader = PyPDFDirectoryLoader(DATA_PATH)

    # Load all PDF files in the directory 
    pdf_documents = document_loader.load()
    documents.extend(pdf_documents)
    return documents

def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
        documents (list[Document]): List of Document objects containing text content to split.
    Returns:
        list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Size of each chunk in characters
    chunk_overlap=100, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"

def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
        chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """

    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using mxbai-embed embeddings
    db = Chroma.from_documents(
        chunks, # Input documents
        OllamaEmbeddings(
             model="mxbai-embed-large",
        ), # Embedding function
        persist_directory=CHROMA_PATH, # Directory to save the database
        collection_metadata={"hnsw:space": "cosine"}, # Search with cosine similarity
    )

    # Retrieve and print the embeddings for the first document for debugging
    doc = db.get(limit=1, include=["embeddings", "documents"], where={"source": "User input 0"})
    print("Embeddings for the first document:")
    print(f"{doc['embeddings'][0][:3]} ... {doc['embeddings'][0][-3:]}")
    print("Embedding length:", len(doc["embeddings"][0]))
    print("Source document:", doc["documents"][0])

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents() # Load documents from a source
    chunks = split_text(documents) # Split documents into manageable chunks
    save_to_chroma(chunks) # Save the processed data to a data store

generate_data_store()