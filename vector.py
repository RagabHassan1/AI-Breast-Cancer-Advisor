from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import time

# Add debug prints
print("Initializing vector store...")

# List your .txt files here
txt_files = ["breast_prevention.txt","self_examination.txt"]

# Check if files exist
for file in txt_files:
    if not os.path.exists(file):
        print(f"WARNING: File {file} does not exist!")

try:
    # Initialize the embeddings model with timeout and retry
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"

    )
    print("Embeddings model initialized")

    # Define Chroma DB location
    db_location = "./chroma_langchain_db"
    add_documents = not os.path.exists(db_location)
    print(f"Vector store location: {db_location}")
    print(f"Need to add documents: {add_documents}")

    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name="text_file_collection",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    print("Vector store initialized")

    # Add documents to the vector store
    if add_documents:
        documents = []
        ids = []

        for file_path in txt_files:
            try:
                if not os.path.exists(file_path):
                    print(f"ERROR: File {file_path} does not exist!")
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                    if not text_content.strip():
                        print(f"WARNING: File {file_path} is empty!")
                        continue
                        
                    documents.append(Document(
                        page_content=text_content,
                        metadata={"source": file_path}
                    ))
                    # Use filename (without .txt) as a unique ID
                    id_str = os.path.splitext(os.path.basename(file_path))[0]
                    ids.append(id_str)
                print(f"Loaded file: {file_path}")
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
        
        if documents:
            print(f"Adding {len(documents)} documents to vector store")
            vector_store.add_documents(documents=documents, ids=ids)
            print("Documents added successfully")
        else:
            print("No documents to add - CRITICAL ERROR")
    
    # Create a retriever with a higher k value to ensure we get results
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Retriever created successfully")
    

except Exception as e:
    print(f"CRITICAL ERROR initializing vector store: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Create a fallback retriever that returns dummy data
    print("Creating fallback retriever...")
    
    class FallbackRetriever:
        def invoke(self, query):
            print(f"Using fallback retriever for query: {query}")
            return [
                Document(
                    page_content="""
                    يتكون كل عضو ف ي جسم الإنسان من أنواع مختلفة من الخلايا،
                    وتنقسم الخلايابشكل طبيع ي. """,
                    metadata={"source": "fallback_data"}
                )
            ]
    
    retriever = FallbackRetriever()