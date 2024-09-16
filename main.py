from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from langchain_community.chat_models import ChatOpenAI # Import OpenAI LLM
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations

# Directory to your pdf files:
DATA_PATH = "data/"
def load_documents():
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.
    Returns:
    List of Document objects: Loaded PDF documents represented as Langchain
                                                            Document objects.
    """
    # Initialize PDF loader with specified directory
    document_loader = PyPDFDirectoryLoader(DATA_PATH) 
    # Load PDF documents and return them as a list of Document objects
    return document_loader.load() 

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

    # Print example of page content and metadata for a chunk
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)

    return chunks # Return the list of split text chunks

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"
LLAMA_PATH = "~/llama"
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

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    OllamaEmbeddings(
       model="llama3.1",
    ),
    persist_directory=CHROMA_PATH
  )

  # Persist the database to disk
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
  """
  Function to generate vector database in chroma from documents.
  """
  documents = load_documents() # Load documents from a source
  chunks = split_text(documents) # Split documents into manageable chunks
  save_to_chroma(chunks) # Save the processed data to a data store

# documents = load_documents() # Call the function
# # Inspect the contents of the first document as well as metadata
# print(documents[0])

generate_data_store()