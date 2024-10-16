from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# The prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

# The query passed into the prompt.
### UPDATE THE PROMPT HERE ###
QUERY_TEXT = "What species of animal are most closely related to llamas?"

CHROMA_PATH = "chroma"
def query_rag(query_text):
    """
    Query your RAG using Chroma and llama.
    Args:
        - query_text (str): The text to query the RAG system with.
    Returns:
        - formatted_response (str): Formatted response including the generated text and sources.
        - response_text (str): The generated response text.
    """

    # For debugging
    print(f"Your query is: {query_text}")

    # YOU MUST - Use same embedding function as in store.py
    embedding_function = OllamaEmbeddings(
             model="mxbai-embed-large",
        )

    # Prepare the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Retrieve the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=1, score_threshold=0.6)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0:
        print(f"Unable to find matching documents. This is a guess.")
    else:
        print(f"Relevance score of returned result: {results[0][1]}")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # For debugging
    print(f"Complete prompt: {prompt}")
    
    # Initialize llama chat model
    model = ChatOllama(
        model="llama3.1",
    )

    # Generate response text based on the prompt
    response_text = model.invoke(prompt)
 
     # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
 
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    return formatted_response, response_text

# Let's call our function we have defined
formatted_response, response_text = query_rag(QUERY_TEXT)
# and finally, inspect our final response!
print(formatted_response)