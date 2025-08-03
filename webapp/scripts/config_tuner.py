#!/usr/bin/env python3
"""
Configuration tuner for adaptive chunking settings.
Allows you to easily experiment with different chunk sizes and strategies.
"""

import sys
from pathlib import Path

# Add the webapp directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import ADAPTIVE_CHUNKING


def show_current_config():
    """Display current adaptive chunking configuration."""
    print("📋 Current Adaptive Chunking Configuration:")
    print("=" * 45)
    for key, value in ADAPTIVE_CHUNKING.items():
        if key == "header_priorities":
            print(f"  {key}:")
            for i, (header, name) in enumerate(value):
                print(f"    {i+1}. {header} ({name})")
        else:
            print(f"  {key}: {value}")


def suggest_settings():
    """Suggest settings for different use cases."""
    print("\n💡 Suggested Settings for Different Use Cases:")
    print("=" * 50)
    
    print("\n🎯 Balanced (recommended default):")
    print("   target_chunk_size: 1500")
    print("   max_chunk_size: 2500") 
    print("   - Good for: Most use cases, balanced context/precision")
    print("   - Pros: Good context without being too large")
    print("   - Cons: None significant")
    
    print("\n⚡ Precision-focused:")
    print("   target_chunk_size: 1000")
    print("   max_chunk_size: 1800")
    print("   - Good for: Quick answers, specific functions")
    print("   - Pros: More precise matching, faster retrieval")
    print("   - Cons: May lack broader context")
    
    print("\n📚 Context-rich (use with caution):")
    print("   target_chunk_size: 2000")
    print("   max_chunk_size: 3000")
    print("   - Good for: Complex procedures, tutorial content")
    print("   - Pros: Maximum context per chunk")
    print("   - Cons: May retrieve less relevant info, slower processing")
    
    print("\n🔧 Code-heavy documentation:")
    print("   target_chunk_size: 1200")
    print("   max_chunk_size: 2200")
    print("   size_threshold_multiplier: 1.2")
    print("   - Good for: API docs with code examples")
    print("   - Pros: Keeps code examples intact")
    
    print("\n⚠️  SAFETY GUIDELINES:")
    print("   • Keep max_chunk_size under 3000 for most embeddings")
    print("   • Ensure max_chunk_size > target_chunk_size * 1.3")
    print("   • Very large chunks (>4000) can hurt retrieval quality")
    print("   • Test with 'make analyze-chunks' after changes")


def interactive_tuner():
    """Interactive configuration tuner."""
    print("\n🔧 Interactive Configuration Tuner")
    print("=" * 35)
    print("(Press Enter to keep current value)")
    
    current = ADAPTIVE_CHUNKING.copy()
    
    # Target chunk size
    while True:
        try:
            response = input(f"\nTarget chunk size ({current['target_chunk_size']}): ").strip()
            if not response:
                break
            target = int(response)
            if 500 <= target <= 3000:
                current['target_chunk_size'] = target
                break
            else:
                print("⚠️  Recommended range: 500-3000 characters")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Max chunk size
    while True:
        try:
            response = input(f"Max chunk size ({current['max_chunk_size']}): ").strip()
            if not response:
                break
            max_size = int(response)
            if max_size <= current['target_chunk_size']:
                print(f"❌ Max size must be larger than target size ({current['target_chunk_size']})")
                continue
            if max_size > 5000:
                print("⚠️  Warning: Very large chunks (>5000) may hurt retrieval quality")
                confirm = input("Continue anyway? (y/N): ").strip().lower()
                if confirm != 'y':
                    continue
            current['max_chunk_size'] = max_size
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Size threshold multiplier
    while True:
        try:
            response = input(f"Size threshold multiplier ({current['size_threshold_multiplier']}): ").strip()
            if not response:
                break
            multiplier = float(response)
            if 1.0 <= multiplier <= 3.0:
                current['size_threshold_multiplier'] = multiplier
                break
            else:
                print("⚠️  Recommended range: 1.0-3.0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print("\n📝 New configuration:")
    for key, value in current.items():
        if key != "header_priorities":
            print(f"  {key}: {value}")
    
    # Validation warnings
    ratio = current['max_chunk_size'] / current['target_chunk_size']
    if ratio < 1.3:
        print("\n⚠️  WARNING: Max size should be at least 1.3x target size for flexibility")
    if current['max_chunk_size'] > 4000:
        print("\n⚠️  WARNING: Very large max size may impact retrieval quality")
    
    print("\n💾 To apply these settings, update config/settings.py:")
    print("ADAPTIVE_CHUNKING = {")
    for key, value in current.items():
        if key == "header_priorities":
            continue
        print(f'    "{key}": {value},')
    print('    "header_priorities": [')
    for header, name in current["header_priorities"]:
        print(f'        ("{header}", "{name}"),')
    print('    ]')
    print("}")
    
    print(f"\n🔍 After updating, run 'make analyze-chunks' to see the impact!")


def main():
    """Main configuration tool."""
    print("⚙️  SLEAP Documentation RAG - Adaptive Chunking Configuration")
    print("=" * 65)
    
    while True:
        print("\n📋 Options:")
        print("1. Show current configuration")
        print("2. Show suggested settings")
        print("3. Interactive tuner")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            show_current_config()
        elif choice == "2":
            suggest_settings()
        elif choice == "3":
            interactive_tuner()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
