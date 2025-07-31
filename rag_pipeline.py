import os
import dotenv
import chromadb
from typing import List, Dict, Optional, Protocol
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from tqdm import tqdm


def setup_environment():
    """Loads environment variables from .local.env and sets up LangSmith."""
    dotenv.load_dotenv(".local.env")
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    # Ensure LANGSMITH_API_KEY is set, if not already in the environment
    if "LANGSMITH_API_KEY" not in os.environ:
        api_key = dotenv.get_key(".local.env", "LANGSMITH_API_KEY")
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = "rag-sleap-docs"
    print("Environment setup for LangSmith tracing.")


class DocumentProcessor:
    """Handles loading, filtering, and transforming of documents."""

    def __init__(self, exclude_patterns: List[str] = None):
        self.exclude_patterns = exclude_patterns or ["/develop/", "genindex.html", "modindex.html", "search.html"]

    def load_html_docs(self, dir_path: str) -> List[Document]:
        """Loads HTML documents from a directory."""
        print(f"Loading documents from: {dir_path}")
        loader = DirectoryLoader(
            dir_path,
            glob="**/*.html",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        return loader.load()

    def filter_docs(self, docs: List[Document]) -> List[Document]:
        """Filters documents based on source path."""
        filtered = []
        for doc in docs:
            source_path = doc.metadata.get('source', '')
            if not any(pattern in source_path for pattern in self.exclude_patterns):
                filtered.append(doc)
        print(f"Filtered {len(docs)} -> {len(filtered)} documents")
        return filtered

    def _get_main_content_selector(self, html_content: str) -> str:
        """Invokes an LLM to find the best CSS selector for the main content."""
        print("\nAsking LLM to find the best selector...")
        llm = ChatVertexAI(model_name="gemini-1.5-flash-001", temperature=0)
        prompt_template = ChatPromptTemplate.from_template(
            """Analyze the following HTML and identify the best CSS selector for the main content area.
            Respond with ONLY the CSS selector string.

            HTML:
            {html_content}"""
        )
        chain = prompt_template | llm | StrOutputParser()
        selector = chain.invoke({"html_content": html_content})
        print(f"✅ LLM identified selector: '{selector.strip()}'")
        return selector.strip()

    def transform_docs(self, docs: List[Document]) -> List[Document]:
        """Extracts main content from HTML documents."""
        if not docs:
            return []

        print(f"\nTransforming {len(docs)} documents...")
        sample_html = docs[0].page_content
        selector = self._get_main_content_selector(sample_html)

        transformed_docs = []
        for doc in tqdm(docs, desc="Parsing HTML content"):
            soup = BeautifulSoup(doc.page_content, "html.parser")
            main_content = soup.select_one(selector)
            clean_content = main_content.get_text(separator=' ', strip=True) if main_content else ""
            transformed_docs.append(Document(page_content=clean_content, metadata=doc.metadata))
        return transformed_docs

    def split_documents(self, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Splits documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        splits = text_splitter.split_documents(docs)
        print(f"Split {len(docs)} documents into {len(splits)} chunks.")
        return splits


class VectorStoreManager:
    """Manages the ChromaDB vector store."""

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
        print("ChromaDB client initialized.")

    def get_or_create_vectorstore(self, collection_name: str, documents: Optional[List[Document]] = None) -> Chroma:
        """Gets an existing vector store or creates a new one from documents."""
        collection = self.client.get_or_create_collection(name=collection_name)
        
        if collection.count() > 0:
            print(f"Found existing collection '{collection_name}' with {collection.count()} documents.")
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
        elif documents:
            print(f"Creating new collection '{collection_name}' and embedding {len(documents)} documents...")
            vectorstore = Chroma.from_documents(
                documents,
                self.embeddings,
                collection_name=collection_name,
                client=self.client,
            )
        else:
            raise ValueError("Documents must be provided to create a new collection.")
            
        return vectorstore


class RAGStrategy(Protocol):
    """A protocol defining the interface for a RAG strategy."""
    def invoke(self, question: str) -> str:
        ...

class BasicRAG(RAGStrategy):
    """Implements a basic RAG pipeline."""

    def __init__(self, retriever, llm_model_name: str = "gemini-1.5-flash-001"):
        self.retriever = retriever
        self.llm = ChatVertexAI(model_name=llm_model_name, temperature=0.2)
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()

    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
You are a helpful AI assistant specialized in SLEAP and SLEAP-IO documentation.
Use the following context to answer the user's question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:
""")

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_chain(self):
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> str:
        """Invokes the RAG chain with a question."""
        print(f"\nInvoking Basic RAG chain with question: '{question}'")
        return self.chain.invoke(question)


def get_rag_pipeline(strategy_name: str, retriever) -> RAGStrategy:
    """Factory function to get a RAG pipeline based on the strategy name."""
    if strategy_name == "basic":
        return BasicRAG(retriever)
    # Add other strategies here like "hyde", "multiquery", etc.
    # elif strategy_name == "hyde":
    #     return HyDERAG(retriever) 
    else:
        raise ValueError(f"Unknown RAG strategy: {strategy_name}")


def main():
    """Main function to run the RAG pipeline."""
    setup_environment()

    # --- Configuration ---
    SLEAP_DOCS_URL = "./sleap_docs"
    SLEAP_IO_DOCS_URL = "./sleap_io_docs/0.4.0"
    SLEAP_COLLECTION = "sleap"
    SLEAP_IO_COLLECTION = "sleap_io"

    # --- Processing ---
    doc_processor = DocumentProcessor()
    vector_manager = VectorStoreManager()

    # Process SLEAP docs
    sleap_docs_raw = doc_processor.load_html_docs(SLEAP_DOCS_URL)
    sleap_docs_filtered = doc_processor.filter_docs(sleap_docs_raw)
    sleap_docs_transformed = doc_processor.transform_docs(sleap_docs_filtered)
    sleap_splits = doc_processor.split_documents(sleap_docs_transformed)
    
    # Process SLEAP-IO docs
    sleap_io_docs_raw = doc_processor.load_html_docs(SLEAP_IO_DOCS_URL)
    sleap_io_docs_filtered = doc_processor.filter_docs(sleap_io_docs_raw)
    sleap_io_docs_transformed = doc_processor.transform_docs(sleap_io_docs_filtered)
    sleap_io_splits = doc_processor.split_documents(sleap_io_docs_transformed)

    # --- Vector Store ---
    sleap_vectorstore = vector_manager.get_or_create_vectorstore(SLEAP_COLLECTION, sleap_splits)
    # sleap_io_vectorstore = vector_manager.get_or_create_vectorstore(SLEAP_IO_COLLECTION, sleap_io_splits)
    
    # For this example, we'll just use the SLEAP retriever
    sleap_retriever = sleap_vectorstore.as_retriever(search_kwargs={"k": 5})

    # --- RAG Pipeline ---
    rag_pipeline = get_rag_pipeline("basic", sleap_retriever)
    
    # --- Querying ---
    question = "How do I fine-tune an existing model with new data in SLEAP?"
    answer = rag_pipeline.invoke(question)
    
    print("\n--- Final Answer ---")
    print(answer)
    print("--------------------")


if __name__ == "__main__":
    main()
