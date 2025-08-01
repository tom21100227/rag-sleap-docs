from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
import chromadb
from config.settings import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    HEADERS_TO_SPLIT_ON,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)


class DocumentSplitter:
    """Handles document splitting and chunking."""
    
    def __init__(self):
        # For splitting guides by their headers
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        
        # For splitting very long docstrings OR language-specific files
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # For RST files
        self.rst_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.RST, 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents based on their type and content."""
        final_chunks = []
        
        for doc in documents:
            source_type = doc.metadata.get("source", "")
            file_path = doc.metadata.get("file", "")

            # A. Process structured guides
            if "guide" in source_type:
                if file_path.endswith(".md"):
                    chunks = self.md_splitter.split_text(doc.page_content)
                    # Add original metadata back to the new chunks
                    for chunk in chunks:
                        chunk.metadata.update(doc.metadata)
                    final_chunks.extend(chunks)

                elif file_path.endswith(".rst"):
                    # Use the language-specific splitter for RST
                    chunks = self.rst_splitter.split_documents([doc])
                    final_chunks.extend(chunks)

            # B. Process code docstrings
            elif "api" in source_type:
                # If the docstring is larger than our chunk size, split it
                if len(doc.page_content) > CHUNK_SIZE:
                    chunks = self.recursive_splitter.split_documents([doc])
                    final_chunks.extend(chunks)
                else:
                    # Otherwise, keep it as a single, logical chunk
                    final_chunks.append(doc)
        
        print(f"Total documents processed: {len(documents)}")
        print(f"Total chunks created: {len(final_chunks)}")
        return final_chunks


class EmbeddingManager:
    """Manages vector embeddings and ChromaDB operations."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
    
    def initialize_vectorstore(self, documents: List[Document]) -> None:
        """Initialize or load the vector store with documents."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            
            if collection.count() > 0:
                print(f"Collection already has {collection.count()} documents. Loading existing vectorstore.")
                self.vectorstore = Chroma(
                    client=self.client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embeddings
                )
            else:
                print("Embedding documents...")
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    collection_name=COLLECTION_NAME,
                    client=self.client,
                )
        except Exception:
            # Collection doesn't exist, create it
            print("Creating new collection and embedding documents...")  
            self.vectorstore = Chroma.from_documents(
                documents,
                self.embeddings,
                collection_name=COLLECTION_NAME,
                client=self.client,
            )
    
    def get_retriever(self, k: int = 6):
        """Get a retriever for the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call initialize_vectorstore first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
