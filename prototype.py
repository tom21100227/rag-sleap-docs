import os
import dotenv
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = dotenv.get_key(".local.env", "LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "rag-sleap-docs"

# %%
import langchain
import langchain_community
import langchain_google_vertexai
import chromadb
import os

# %%
# Sanity check: at this point I should be able to see the LangSmith configuration
from langchain_google_vertexai import ChatVertexAI

# gemini = ChatVertexAI(
#     model_name="gemini-2.0-flash-lite",
#     temperature=0.2)
# gemini.invoke("Repeat after me: 'Hello, world!'")

# %%
# Make or bind a ChromaDB client

client = chromadb.PersistentClient(path="./chroma_db")
try:
    sleap_collection = client.get_or_create_collection(name="sleap")
    sleap_io_collection = client.get_or_create_collection(name="sleap_io")
    print("Collections created or accessed successfully.")
except Exception as e:
    print(f"Error creating or accessing collection: {e}")

# %%
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma

SLEAP_DOCS_URL = "./sleap_docs"
SLEAP_IO_DOCS_URL = "./sleap_io_docs/0.4.0"

# %%
sleap_loader = DirectoryLoader(
    SLEAP_DOCS_URL,
    glob="**/*.html",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

sleap_io_loader = DirectoryLoader(
    SLEAP_IO_DOCS_URL,
    glob="**/*.html",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

# %%
sleap_docs = sleap_loader.load()
sleap_io_docs = sleap_io_loader.load()

# %%
# Filter out unwanted documents
def filter_docs(docs):
    filtered = []
    exclude_patterns = ["/develop/", "genindex.html", "modindex.html", "search.html"]
    
    for doc in docs:
        source_path = doc.metadata.get('source', '')
        if not any(pattern in source_path for pattern in exclude_patterns):
            filtered.append(doc)
        else:
            print(f"Excluding: {source_path}")
    
    print(f"Filtered {len(docs)} -> {len(filtered)} documents")
    return filtered

sleap_docs = filter_docs(sleap_docs)
sleap_io_docs = filter_docs(sleap_io_docs)

# %%
import os
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from tqdm import tqdm


def get_main_content_selector(html_content: str) -> str:
    """
    Invokes an LLM to find the best CSS selector for the main content
    of a given HTML document. This is the "expert" we call when needed.
    """
    print("\nAsking LLM to find the best selector for a specific page...")
    
    # Initialize the LLM
    llm = ChatVertexAI(model_name="gemini-2.0-flash-lite", temperature=0)
    
    # Define the prompt for the LLM
    prompt_template = ChatPromptTemplate.from_template(
        """Analyze the following HTML document and identify the single best CSS selector 
that precisely targets the main article or content area.
Ignore headers, footers, navigation bars, and sidebars.
Respond with ONLY the CSS selector string and nothing else. Do NOT include any additional text or explanations. Do NOT include a markdown code block.

HTML:
{html_content}
"""
    )
    
    # Define the chain to get the selector
    chain = prompt_template | llm | StrOutputParser()
    
    # Invoke the chain
    selector = chain.invoke({"html_content": html_content})
    
    print(f"✅ LLM identified new selector: '{selector.strip()}'")
    return selector.strip()

def transform_docs_with_selector(docs: list[Document], selectors: list[str]) -> list[Document]:
    """
    Applies a list of CSS selectors to a list of documents.
    If no selectors work for a document, it calls an LLM to find a new one
    and adds it to the list for future use.
    """
    print(f"\nApplying selectors to all {len(docs)} documents...")
    transformed_docs = []
    selector_counts = {}  # Track usage of each selector

    for doc in tqdm(docs, desc="Processing documents", unit="doc"):
        soup = BeautifulSoup(doc.page_content, "html.parser")
        
        clean_content = ""
        content_found = False
        used_selector = None
        
        # Try all existing selectors first
        for selector in selectors:
            main_content_element = soup.select_one(selector)
            if main_content_element:
                clean_content = main_content_element.get_text(separator=' ', strip=True)
                content_found = True
                used_selector = selector
                break # Selector worked, move to the next document
        
        # If no existing selectors worked, call the LLM for help
        if not content_found:
            print(f"    ⚠️ No existing selectors worked for {doc.metadata['source']}. Finding a new one...")
            new_selector = get_main_content_selector(doc.page_content)
            selectors.append(new_selector) # Add the new selector to our list
            used_selector = new_selector
            
            # Try again with the new selector
            main_content_element = soup.select_one(new_selector)
            if main_content_element:
                clean_content = main_content_element.get_text(separator=' ', strip=True)
            else:
                clean_content = f"[Content not found even with new selector '{new_selector}']"

        # Track selector usage
        if used_selector:
            selector_counts[used_selector] = selector_counts.get(used_selector, 0) + 1

        transformed_docs.append(Document(page_content=clean_content, metadata=doc.metadata))

    # Print selector usage statistics
    print("\n📊 Selector Usage Statistics:")
    print("-" * 50)
    for selector, count in sorted(selector_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(docs)) * 100
        print(f"'{selector}': {count} docs ({percentage:.1f}%)")
    
    print(f"\nTotal selectors used: {len(selector_counts)}")
    print(f"Total documents processed: {len(docs)}")
        
    return transformed_docs

# %%


# Transform raw HTML documents using LLM-based selector detection
print("\n=== Processing SLEAP docs with LLM selector detection ===")
if sleap_docs:
    # Get the selector by analyzing the FIRST document
    sample_html = sleap_docs[0].page_content
    sleap_selector = get_main_content_selector(sample_html)
    
    # Apply that selector to ALL sleap documents
    sleap_docs = transform_docs_with_selector(sleap_docs, [sleap_selector])

print("\n=== Processing SLEAP-IO docs with LLM selector detection ===")
if sleap_io_docs:
    # Get the selector by analyzing the FIRST document  
    sample_html = sleap_io_docs[0].page_content
    sleap_io_selector = get_main_content_selector(sample_html)
    
    # Apply that selector to ALL sleap-io documents
    sleap_io_docs = transform_docs_with_selector(sleap_io_docs, [sleap_io_selector])

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust chunk size as needed
    chunk_overlap=200,  # Adjust overlap as needed
    length_function=len)
sleap_splits = text_splitter.split_documents(sleap_docs)
sleap_io_splits = text_splitter.split_documents(sleap_io_docs)

# %%
# Now I embed the documents and add them to the ChromaDB collections
from langchain_google_vertexai import VertexAIEmbeddings
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004"
)

# Check if collections already have data
sleap_collection = client.get_collection("sleap")
sleap_io_collection = client.get_collection("sleap_io")

if sleap_collection.count() > 0:
    print(f"SLEAP collection already has {sleap_collection.count()} documents. Skipping embedding.")
    sleap_vectorstore = Chroma(
        client=client,
        collection_name="sleap",
        embedding_function=embeddings
    )
else:
    print("Embedding SLEAP documents...")
    sleap_vectorstore = Chroma.from_documents(
        sleap_splits,
        embeddings,
        collection_name="sleap",
        client=client,
    )

if sleap_io_collection.count() > 0:
    print(f"SLEAP-IO collection already has {sleap_io_collection.count()} documents. Skipping embedding.")
    sleap_io_vectorstore = Chroma(
        client=client,
        collection_name="sleap_io",
        embedding_function=embeddings
    )
else:
    print("Embedding SLEAP-IO documents...")
    sleap_io_vectorstore = Chroma.from_documents(
        sleap_io_splits,
        embeddings,
        collection_name="sleap_io",
        client=client,
    )

# %%
sleap_retriever = sleap_vectorstore.as_retriever()
sleap_io_retriever = sleap_io_vectorstore.as_retriever()

# %%
# Test this out
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant specialized in SLEAP (Social LEAP Estimates Animal Poses) and SLEAP-IO documentation.

Use the following context from the documentation to answer the user's question. If the answer cannot be found in the context, say "I don't have enough information in the provided documentation to answer that question."

Context:
{context}

Question: {question}

Instructions:
- Provide accurate, detailed answers based on the documentation
- Include code examples when relevant
- Mention specific function names, classes, or modules when applicable
- If discussing installation or setup, be specific about requirements
- For troubleshooting questions, provide step-by-step solutions
- Always cite which part of the documentation (SLEAP or SLEAP-IO) your answer comes from

Answer:
""")

llm = ChatVertexAI(
    model_name="gemini-2.0-flash-lite",
    temperature=0.2)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": sleap_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("How do I fine-tune an existing model with new data in SLEAP?")

# %%



