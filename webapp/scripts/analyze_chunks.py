#!/usr/bin/env python3
"""
Tool to analyze and compare chunking strategies for the SLEAP documentation.
"""

import sys
from pathlib import Path
import statistics

# Add the webapp directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_parser import DocumentParser
from src.embeddings import AdaptiveDocumentSplitter
from config.settings import REPO_PATHS


def analyze_chunks(chunks, name="Chunks"):
    """Analyze chunk statistics."""
    if not chunks:
        print(f"❌ No {name.lower()} found")
        return
    
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    print(f"\n📊 {name} Analysis:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Average size: {statistics.mean(sizes):.0f} chars")
    print(f"   Median size: {statistics.median(sizes):.0f} chars")
    print(f"   Size range: {min(sizes)} - {max(sizes)} chars")
    print(f"   Standard deviation: {statistics.stdev(sizes):.0f} chars")
    
    # Size distribution with more granular buckets
    tiny = len([s for s in sizes if s < 300])
    small = len([s for s in sizes if 300 <= s < 800])
    medium = len([s for s in sizes if 800 <= s < 1500])
    large = len([s for s in sizes if 1500 <= s < 2500])
    xlarge = len([s for s in sizes if 2500 <= s < 4000])
    huge = len([s for s in sizes if s >= 4000])
    
    print(f"   Size distribution:")
    print(f"     Tiny (<300): {tiny} ({tiny/len(chunks)*100:.1f}%)")
    print(f"     Small (300-800): {small} ({small/len(chunks)*100:.1f}%)")
    print(f"     Medium (800-1500): {medium} ({medium/len(chunks)*100:.1f}%)")
    print(f"     Large (1500-2500): {large} ({large/len(chunks)*100:.1f}%)")
    print(f"     X-Large (2500-4000): {xlarge} ({xlarge/len(chunks)*100:.1f}%)")
    print(f"     Huge (4000+): {huge} ({huge/len(chunks)*100:.1f}%)")
    
    # Warnings
    if huge > 0:
        print(f"   ⚠️  WARNING: {huge} chunks are very large (4000+ chars) - may impact retrieval quality")
    if tiny > len(chunks) * 0.2:
        print(f"   ⚠️  WARNING: {tiny/len(chunks)*100:.1f}% of chunks are very small - consider better combining")
    
    # Quality score
    ideal_range = len([s for s in sizes if 800 <= s <= 2500])
    quality_score = ideal_range / len(chunks) * 100
    print(f"   Quality score: {quality_score:.1f}% of chunks in ideal range (800-2500 chars)")
    
    if quality_score < 60:
        print(f"   💡 Consider tuning adaptive chunking settings for better size distribution")


def show_sample_chunks(chunks, name="Chunks", num_samples=3):
    """Show sample chunks."""
    print(f"\n📄 Sample {name}:")
    
    # Show samples of different sizes
    sizes = [(i, len(chunk.page_content)) for i, chunk in enumerate(chunks)]
    sizes.sort(key=lambda x: x[1])
    
    # Get small, medium, and large samples
    indices = [
        sizes[len(sizes)//4][0],   # 25th percentile
        sizes[len(sizes)//2][0],   # Median
        sizes[3*len(sizes)//4][0]  # 75th percentile
    ]
    
    for i, idx in enumerate(indices[:num_samples]):
        chunk = chunks[idx]
        size = len(chunk.page_content)
        source = chunk.metadata.get('source', 'unknown')
        file_name = chunk.metadata.get('file', 'unknown')
        
        print(f"\n--- Sample {i+1} ({size} chars, {source}, {file_name}) ---")
        preview = chunk.page_content[:200].replace('\n', ' ')
        print(f"{preview}...")


def main():
    """Analyze chunking strategies."""
    print("🔍 SLEAP Documentation Chunking Analysis")
    print("=" * 50)
    
    # Step 1: Parse documents
    print("\n📚 Step 1: Parsing documents...")
    parser = DocumentParser(REPO_PATHS)
    raw_documents = parser.parse_all_repositories()
    
    if not raw_documents:
        print("❌ No documents found. Please check your repository paths.")
        return
    
    print(f"✅ Found {len(raw_documents)} raw documents")
    
    # Step 2: Test adaptive chunking
    print("\n🔧 Step 2: Testing adaptive chunking...")
    adaptive_splitter = AdaptiveDocumentSplitter()
    adaptive_chunks = adaptive_splitter.split_documents(raw_documents)
    
    # Step 3: Analysis
    analyze_chunks(adaptive_chunks, "Adaptive Chunks")
    
    # Step 4: Show samples
    show_sample_chunks(adaptive_chunks, "Adaptive Chunks")
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💡 Key benefits of adaptive chunking:")
    print(f"   • Larger, more contextual chunks (target: 2000 chars)")
    print(f"   • Smart header-level selection based on content size")
    print(f"   • Combines small related sections")
    print(f"   • Preserves semantic coherence")
    print(f"   • Better retrieval context for RAG")


if __name__ == "__main__":
    main()
