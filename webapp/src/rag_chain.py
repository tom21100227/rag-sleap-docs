from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
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
        
        # Build the conversational RAG chain
        self._build_chain()
    
    def _format_docs(self, docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _get_chat_history(self, _):
        """Get chat history from memory."""
        return self.memory.get_chat_history()
    
    def _build_chain(self):
        """Build the conversational RAG chain."""
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(self._get_chat_history),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _generate_queries(self, question: str, method: str) -> List[str]:
        """Generate multiple queries based on the translation method."""
        if method == "none":
            return [question]
        
        prompt_map = {
            "multi_query": MULTI_QUERY_PROMPT,
            "rag_fusion": RAG_FUSION_PROMPT,
            "decomposition": DECOMPOSITION_PROMPT,
            "step_back": STEP_BACK_PROMPT
        }
        
        if method not in prompt_map:
            return [question]
        
        prompt = ChatPromptTemplate.from_template(prompt_map[method])
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({"question": question})
        
        # Split the result into individual queries
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        
        # Always include the original question
        if question not in queries:
            queries.insert(0, question)
            
        return queries
    
    def _apply_hyde(self, queries: List[str]) -> List[str]:
        """Apply HyDE (Hypothetical Document Embeddings) to queries."""
        hyde_prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        
        hyde_docs = []
        for query in queries:
            try:
                hyde_doc = hyde_chain.invoke({"question": query})
                hyde_docs.append(hyde_doc)
            except Exception as e:
                # Fallback to original query if HyDE fails
                hyde_docs.append(query)
        
        return hyde_docs
    
    def _retrieve_with_fusion(self, queries: List[str], k: int = 6) -> List:
        """Retrieve documents using multiple queries and apply reciprocal rank fusion."""
        all_docs = []
        doc_scores = {}
        
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            for rank, doc in enumerate(docs[:k]):
                doc_key = doc.page_content[:100]  # Use content snippet as key
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0}
                
                # Reciprocal rank fusion: 1/(rank + 60)
                doc_scores[doc_key]["score"] += 1.0 / (rank + 60)
        
        # Sort by fusion score and return top k documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:k]]
    
    def chat_with_memory(self, question: str, query_method: str = "none", use_hyde: bool = False) -> dict:
        """Process a question with query translation and return both answer and retrieved documents."""
        
        # Generate multiple queries based on the method
        queries = self._generate_queries(question, query_method)
        
        # Apply HyDE if requested
        if use_hyde:
            queries = self._apply_hyde(queries)
        
        # Retrieve documents
        if query_method == "rag_fusion" or len(queries) > 1:
            # Use fusion for multiple queries
            retrieved_docs = self._retrieve_with_fusion(queries)
        else:
            # Standard retrieval for single query
            retrieved_docs = self.retriever.get_relevant_documents(queries[0])
        
        # Format context and get response
        context = self._format_docs(retrieved_docs)
        response = self.chain.invoke(question)
        
        # Save to memory
        self.memory.save_context(question, response)
        
        return {
            "response": response,
            "retrieved_docs": retrieved_docs,
            "generated_queries": queries if query_method != "none" else None
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
        return self.chain.invoke(question)
