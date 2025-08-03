from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
import chromadb
import re
from config.settings import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    HEADERS_TO_SPLIT_ON,
    ADAPTIVE_CHUNKING,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)


class AdaptiveDocumentSplitter:
    """Handles adaptive document splitting and chunking with size-aware header splitting."""
    
    def __init__(self):
        self.config = ADAPTIVE_CHUNKING
        
        # For splitting very long docstrings OR language-specific files
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["target_chunk_size"],
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # For RST files
        self.rst_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.RST, 
            chunk_size=self.config["target_chunk_size"], 
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def _estimate_section_sizes(self, content: str, headers: List[Tuple[str, str]]) -> dict:
        """Estimate the size of sections if split by given headers."""
        if not headers:
            return {"avg_size": len(content), "max_size": len(content), "num_sections": 1}
        
        # Create a pattern to match the headers
        header_pattern = "|".join([re.escape(h[0]) for h in headers])
        sections = re.split(f'^({header_pattern})\\s', content, flags=re.MULTILINE)
        
        # Filter out empty sections
        section_sizes = [len(section) for section in sections if section.strip()]
        
        if not section_sizes:
            return {"avg_size": len(content), "max_size": len(content), "num_sections": 1}
        
        return {
            "avg_size": sum(section_sizes) / len(section_sizes),
            "max_size": max(section_sizes),
            "num_sections": len(section_sizes)
        }
    
    def _get_optimal_header_level(self, content: str) -> List[Tuple[str, str]]:
        """Determine the optimal header level for splitting based on content size."""
        target_size = self.config["target_chunk_size"]
        max_size = self.config["max_chunk_size"]
        
        # Try each header level from largest to smallest
        for i, header_set in enumerate([
            [("#", "H1")],                              # Try H1 only
            [("#", "H1"), ("##", "H2")],               # Try H1 and H2
            [("#", "H1"), ("##", "H2"), ("###", "H3")], # Try H1, H2, H3
            self.config["header_priorities"][:4],       # Try up to H4
            self.config["header_priorities"]            # Try all headers
        ]):
            stats = self._estimate_section_sizes(content, header_set)
            
            # If average size is close to target and max isn't too big, use this level
            if (stats["avg_size"] <= target_size * self.config["size_threshold_multiplier"] and 
                stats["max_size"] <= max_size):
                print(f"  -> Using header level {i+1} (avg: {stats['avg_size']:.0f}, max: {stats['max_size']:.0f}, sections: {stats['num_sections']})")
                return header_set
        
        # If no header level works well, use all headers (finest granularity)
        print(f"  -> Using finest granularity (all headers)")
        return self.config["header_priorities"]
    
    def _smart_combine_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """Combine small adjacent chunks to reach target size, but respect max_chunk_size."""
        if not chunks:
            return chunks
        
        combined_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            current_size = len(current_chunk.page_content)
            next_size = len(next_chunk.page_content)
            combined_size = current_size + next_size + 2  # +2 for "\n\n" separator
            
            # More conservative combining: only if result stays well under max_chunk_size
            # and current chunk is significantly smaller than target
            if (combined_size <= self.config["max_chunk_size"] * 0.8 and  # Stay under 80% of max
                current_size < self.config["target_chunk_size"] * 0.7 and  # Current is less than 70% of target
                current_chunk.metadata.get("file") == next_chunk.metadata.get("file")):
                
                # Combine the chunks
                current_chunk.page_content += "\n\n" + next_chunk.page_content
                # Keep metadata from first chunk, but note it's combined
                current_chunk.metadata["combined"] = True
            else:
                # Can't combine safely, add current chunk and start new one
                combined_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # Don't forget the last chunk
        combined_chunks.append(current_chunk)
        
        return combined_chunks
    
    def _split_markdown_adaptively(self, content: str, metadata: dict) -> List[Document]:
        """Split markdown content using adaptive header-based splitting."""
        print(f"  Adaptively splitting markdown: {metadata.get('file', 'unknown')} ({len(content)} chars)")
        
        # If content is already small enough, don't split
        if len(content) <= self.config["target_chunk_size"]:
            return [Document(page_content=content, metadata=metadata)]
        
        # Find optimal header level for this content
        optimal_headers = self._get_optimal_header_level(content)
        
        # Create splitter with optimal headers
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=optimal_headers,
            strip_headers=False
        )
        
        try:
            chunks = md_splitter.split_text(content)
            # Add original metadata back to the new chunks
            for chunk in chunks:
                chunk.metadata.update(metadata)
            
            # Try to combine small chunks
            chunks = self._smart_combine_small_chunks(chunks)
            
            print(f"    -> Created {len(chunks)} adaptive chunks")
            return chunks
            
        except Exception as e:
            print(f"    -> Header splitting failed: {e}, falling back to recursive splitting")
            # Fallback to recursive splitting
            doc = Document(page_content=content, metadata=metadata)
            return self.recursive_splitter.split_documents([doc])
    
    def _apply_safety_splitting(self, chunks: List[Document]) -> List[Document]:
        """Apply recursive splitting to any chunks that exceed max_chunk_size as a safety net."""
        safe_chunks = []
        oversized_count = 0
        
        for chunk in chunks:
            if len(chunk.page_content) > self.config["max_chunk_size"]:
                oversized_count += 1
                # Apply recursive splitting to oversized chunks
                split_chunks = self.recursive_splitter.split_documents([chunk])
                safe_chunks.extend(split_chunks)
                print(f"    -> Safety split oversized chunk ({len(chunk.page_content)} chars) into {len(split_chunks)} pieces")
            else:
                safe_chunks.append(chunk)
        
        if oversized_count > 0:
            print(f"  -> Applied safety splitting to {oversized_count} oversized chunks")
        
        return safe_chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using adaptive chunking strategy."""
        final_chunks = []
        
        print(f"\n🔄 Starting adaptive document splitting...")
        print(f"Target chunk size: {self.config['target_chunk_size']}, Max: {self.config['max_chunk_size']}")
        
        for i, doc in enumerate(documents):
            source_type = doc.metadata.get("source", "")
            file_path = doc.metadata.get("file", "")
            
            if i % 10 == 0:  # Progress indicator
                print(f"Processing document {i+1}/{len(documents)}")

            # A. Process structured guides
            if "guide" in source_type:
                if file_path.endswith(".md"):
                    chunks = self._split_markdown_adaptively(doc.page_content, doc.metadata)
                    final_chunks.extend(chunks)

                elif file_path.endswith(".rst"):
                    # For RST, check size and split if necessary
                    if len(doc.page_content) > self.config["max_chunk_size"]:
                        chunks = self.rst_splitter.split_documents([doc])
                        final_chunks.extend(chunks)
                    else:
                        final_chunks.append(doc)

            # B. Process code docstrings - keep them larger when possible
            elif "api" in source_type:
                # Only split if significantly larger than max size
                if len(doc.page_content) > self.config["max_chunk_size"]:
                    chunks = self.recursive_splitter.split_documents([doc])
                    final_chunks.extend(chunks)
                else:
                    # Keep docstrings intact - they're usually coherent units
                    final_chunks.append(doc)
        
        # Apply safety splitting to ensure no chunk exceeds max_chunk_size
        print(f"\n🛡️  Applying safety splitting for oversized chunks...")
        final_chunks = self._apply_safety_splitting(final_chunks)
        
        # Final statistics
        chunk_sizes = [len(chunk.page_content) for chunk in final_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        oversized = len([s for s in chunk_sizes if s > self.config["max_chunk_size"]])
        
        print(f"\n📊 Adaptive splitting complete:")
        print(f"   Documents processed: {len(documents)}")
        print(f"   Chunks created: {len(final_chunks)}")
        print(f"   Average chunk size: {avg_size:.0f} chars")
        print(f"   Size range: {min(chunk_sizes) if chunk_sizes else 0:.0f} - {max(chunk_sizes) if chunk_sizes else 0:.0f} chars")
        print(f"   Oversized chunks (>{self.config['max_chunk_size']}): {oversized}")
        
        if oversized > 0:
            print(f"   ⚠️  Warning: {oversized} chunks still exceed max size - consider lowering max_chunk_size")
        
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
