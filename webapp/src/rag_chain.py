from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.load import dumps, loads
from config.settings import (
    QUERY_LLM_MODEL, QUERY_LLM_TEMPERATURE, LLM_MODEL, LLM_TEMPERATURE, RAG_PROMPT_TEMPLATE,
    MULTI_QUERY_PROMPT, RAG_FUSION_PROMPT, DECOMPOSITION_PROMPT, 
    STEP_BACK_PROMPT, HYDE_PROMPT
)
import vertexai
from config.settings import GOOGLE_CLOUD_PROJECT_ID
from typing import List, Dict, Optional, Dict, Any
import numpy as np
import threading


class RAGChain:
    """Handles the RAG (Retrieval-Augmented Generation) chain."""
    
    def __init__(self, retriever):
        # Initialize Vertex AI
        vertexai.init(project=GOOGLE_CLOUD_PROJECT_ID)
        
        self.retriever = retriever
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        
        self.query_llm = ChatVertexAI(
            model_name=QUERY_LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Current chat history for this request (passed from outside)
        self._current_chat_history = []
        
        # Threading event for retrieval completion
        self.retrieval_ready = threading.Event()
        
        # Build the query generation chain
        self.generate_queries_chain = (
            ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
            | self.query_llm
            | StrOutputParser()
            | RunnableLambda(self._parse_queries)
        )
        
        # Build the conversational RAG chain
        self._build_chain()

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a single string."""
        return "\n\n".join(str(doc.metadata) + doc.page_content for doc in docs)

    def _get_chat_history(self, _):
        """Get chat history from the current request parameter."""
        if not self._current_chat_history:
            return ""
        
        formatted_history = []
        for message in self._current_chat_history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                formatted_history.append(f"Human: {content}")
            elif role == "assistant":
                formatted_history.append(f"Assistant: {content}")
        
        return "\n".join(formatted_history)
    
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
        
        # Signal that retrieval is complete
        self.retrieval_ready.set()
        
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
            # Return the loaded documents
            return [loads(doc) for doc in unique_docs]

        # Multi-query retrieval logic chain
        multi_query_retrieval_logic_chain = (
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
            multi_query_retrieval_logic_chain
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
        
        def reciprocal_rank_fusion(results: list[list], k=60):
            """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                and an optional parameter k used in the RRF formula """
            
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}

            # Iterate through each list of ranked documents
            for docs in results:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                loads(doc)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]

            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results

        rag_fusion_retrieval_logic_chain = (
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
                    | reciprocal_rank_fusion
                ).with_config({"run_name": "fuse_docs"})
            )
        )

        # Placeholder chains for future methods (not implemented)
        self.rag_fusion_chain = (
            rag_fusion_retrieval_logic_chain
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
        ).with_config({"run_name": "rag_fusion_chain"})
        
        self.decomposition_chain = None
        self.step_back_chain = None
        
        # Storage for retrieved docs and queries (for UI display)
        self._last_retrieved_docs = []
        self._last_generated_queries = []

    def chat_with_memory(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None, query_method: str = "none", use_hyde: bool = False) -> dict:
        """Process a question with query translation and return streaming response with metadata."""
        
        # Set the chat history for this request
        self._current_chat_history = chat_history or []
        
        # HyDE is not implemented yet
        if use_hyde:
            raise NotImplementedError("HyDE is not implemented yet.")
        
        # Reset the state first
        self._last_retrieved_docs = []
        self._last_generated_queries = []
        self.retrieval_ready.clear()  # Reset the threading event
        
        # Select the appropriate chain based on query method
        if query_method == "none":
            chain = self.default_chain
        elif query_method == "multi_query":
            chain = self.multi_query_chain
        elif query_method == "rag_fusion":
            chain = self.rag_fusion_chain
        elif query_method == "decomposition":
            if self.decomposition_chain is None:
                raise NotImplementedError("Decomposition is not implemented yet.")
            chain = self.decomposition_chain
        elif query_method == "step_back":
            if self.step_back_chain is None:
                raise NotImplementedError("Step Back is not implemented yet.")
            chain = self.step_back_chain
        else:
            raise ValueError(f"Unknown query method: {query_method}")
        
        # Stream the response from the selected chain
        response_stream = chain.stream(question)
        
        return {
            "response_stream": response_stream,
            "retrieval_event": self.retrieval_ready
        }

    def chat_without_rag(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> dict:
        """Chat without RAG - just use conversation history."""
        
        # Set the chat history for this request
        self._current_chat_history = chat_history or []
        
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
        ).with_config({
            "run_name": "chat_without_rag_chain"
        })
        
        response_stream = simple_chain.stream(question)
        
        return {
            "response_stream": response_stream,
            "retrieved_docs": [],
            "generated_queries": None
        }

    def chat_without_memory(self, question: str) -> str:
        """Process a question without saving to memory."""
        # Clear chat history for this request
        self._current_chat_history = []
        return self.default_chain.invoke(question)
