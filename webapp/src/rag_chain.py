from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import LLM_MODEL, LLM_TEMPERATURE, RAG_PROMPT_TEMPLATE
from src.memory import ConversationMemory
import vertexai
from config.settings import GOOGLE_CLOUD_PROJECT_ID


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
    
    def chat_with_memory(self, question: str) -> str:
        """Process a question and return an answer, saving to memory."""
        # Get the response
        response = self.chain.invoke(question)
        
        # Save to memory
        self.memory.save_context(question, response)
        
        return response
    
    def chat_without_memory(self, question: str) -> str:
        """Process a question without saving to memory."""
        return self.chain.invoke(question)
