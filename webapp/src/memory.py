from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage
from typing import List


class ConversationMemory:
    """Manages conversation memory for the RAG chain."""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def save_context(self, question: str, answer: str) -> None:
        """Save a question-answer pair to memory."""
        self.memory.save_context({"question": question}, {"answer": answer})
    
    def get_chat_history(self) -> List[BaseMessage]:
        """Get the current chat history."""
        return self.memory.chat_memory.messages
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
