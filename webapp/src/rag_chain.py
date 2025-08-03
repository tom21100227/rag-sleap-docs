from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.load import dumps, loads
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, RAG_PROMPT_TEMPLATE,
    MULTI_QUERY_PROMPT, RAG_FUSION_PROMPT, DECOMPOSITION_PROMPT, 
    STEP_BACK_PROMPT, HYDE_PROMPT
)
from src.memory import ConversationMemory
import vertexai
from config.settings import GOOGLE_CLOUD_PROJECT_ID
from typing import List
import numpy as np


class RAGChain:
    """Handles the RAG (Retrieval-Augmented Generation) chain."""
    
    def __init__(self, retriever, memory: ConversationMemory):
        # Initialize Vertex AI
        vertexai.init(project=GOOGLE_CLOUD_PROJECT_ID)
        
        self.retriever = retriever
        self.memory = memory
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Build the query generation chain
        self.generate_queries_chain = (
            ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
            | self.llm
            | StrOutputParser()
            | RunnableLambda(self._parse_queries)
        )
        
        # Build the conversational RAG chain
        self._build_chain()
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a single string."""
        return "\n\n".join(str(doc.metadata) + doc.page_content for doc in docs)

    def _get_chat_history(self, _):
        """Get chat history from memory."""
        return self.memory.get_chat_history()
    
    def _flatten_docs(self, docs_per_query: List[List[Document]]) -> List[Document]:
        """Flattens the list of lists of documents into a single list."""
        return [doc for sublist in docs_per_query for doc in sublist]
    
    def _save_retrieval_state(self, chain_output: dict) -> dict:
        """
        This function's only job is to save the intermediate state for display.
        It's a passthrough, so it returns its input untouched.
        """
        self._last_retrieved_docs = chain_output.get("retrieved_docs", [])
        self._last_generated_queries = chain_output.get("generated_queries", [])
        return chain_output  # Must return the input to not break the chain
    
    def _parse_queries(self, result: str) -> List[str]:
        """Parse the LLM output into a list of queries."""
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        return queries
    
    def _build_chain(self):
        """Build all the conversational RAG chains."""
        # Default single-query chain with state storage
        default_retrieval_chain = (
            {"question": RunnablePassthrough()}
            | RunnablePassthrough.assign(
                retrieved_docs=lambda x: self.retriever.get_relevant_documents(x["question"]),
                generated_queries=lambda x: None  # No generated queries for default
            )
        )
        
        self.default_chain = (
            default_retrieval_chain
            | RunnableLambda(self._save_retrieval_state)
            | {
                "context": lambda x: self._format_docs(x["retrieved_docs"]),
                "question": lambda x: x["question"],
                "chat_history": RunnableLambda(self._get_chat_history),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        ).with_config({
            "run_name": "default_chain"
        })
        
        def get_unique_union(documents: list[list]):
            """ Unique union of retrieved docs """
            # Flatten list of lists, and convert each Document to string
            flattened_docs = [dumps(doc) for doc in documents]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc) for doc in unique_docs]

        # Multi-query retrieval logic chain
        retrieval_logic_chain = (
            # 1. Start with the question
            {"question": RunnablePassthrough()}
            # 2. Generate queries and add them to the dictionary
            | RunnablePassthrough.assign(
                generated_queries=(lambda x: x["question"]) | self.generate_queries_chain
            ).with_config({"run_name": "generate_queries"})
            # 3. Retrieve docs using the queries and add them to the dictionary
            | RunnablePassthrough.assign(
                retrieved_docs=(
                    (lambda x: x["generated_queries"]) 
                    | self.retriever.map()
                    | self._flatten_docs
                    | get_unique_union
                ).with_config({"run_name": "retrieve_docs"})
            )
        )
        

        # Multi-query chain with observable steps
        self.multi_query_chain = (
            retrieval_logic_chain
            # 4. Handle the side-effect (save state for UI)
            | RunnableLambda(self._save_retrieval_state)
            # 5. Assemble the context for the final prompt
            | {
                "context": lambda x: self._format_docs(x["retrieved_docs"]),
                "question": lambda x: x["question"],
                "chat_history": RunnableLambda(self._get_chat_history),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        ).with_config({"run_name": "multi_query_chain"})

        # Placeholder chains for future methods (not implemented)
        self.rag_fusion_chain = None
        self.decomposition_chain = None
        self.step_back_chain = None
        
        # Storage for retrieved docs and queries (for UI display)
        self._last_retrieved_docs = []
        self._last_generated_queries = []

    def chat_with_memory(self, question: str, query_method: str = "none", use_hyde: bool = False) -> dict:
        """Process a question with query translation and return both answer and retrieved documents."""
        
        # HyDE is not implemented yet
        if use_hyde:
            raise NotImplementedError("HyDE is not implemented yet.")
        
        # Select the appropriate chain - retrieval and storage happens inside the chain
        if query_method == "none":
            chain = self.default_chain
        elif query_method == "multi_query":
            chain = self.multi_query_chain
        elif query_method == "rag_fusion":
            raise NotImplementedError("RAG Fusion is not implemented yet.")
        elif query_method == "decomposition":
            raise NotImplementedError("Decomposition is not implemented yet.")
        elif query_method == "step_back":
            raise NotImplementedError("Step Back is not implemented yet.")
        else:
            raise ValueError(f"Unknown query method: {query_method}")
        
        # Get response from the selected chain (this also populates _last_retrieved_docs and _last_generated_queries)
        response = chain.invoke(question)
        
        # Save to memory
        self.memory.save_context(question, response)
        
        return {
            "response": response,
            "retrieved_docs": self._last_retrieved_docs,
            "generated_queries": self._last_generated_queries
        }
    
    def chat_without_rag(self, question: str) -> dict:
        """Chat without RAG - just use conversation memory."""
        # Simple conversation prompt without context
        simple_prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant specialized in SLEAP (Social LEAP Estimates Animal Poses), SLEAP-IO, and DREEM.
        
        Chat History:
        {chat_history}
        
        Current Question: {question}
        
        Please continue the conversation. If the question is about SLEAP, SLEAP-IO, or DREEM but you need specific documentation details, mention that you would need to search the documentation for accurate details.
        
        Answer:
        """)
        
        simple_chain = (
            {
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(self._get_chat_history),
            }
            | simple_prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = simple_chain.invoke(question)
        
        # Save to memory
        self.memory.save_context(question, response)
        
        return {
            "response": response,
            "retrieved_docs": [],
            "generated_queries": None
        }
    
    def chat_without_memory(self, question: str) -> str:
        """Process a question without saving to memory."""
        return self.default_chain.invoke(question)
