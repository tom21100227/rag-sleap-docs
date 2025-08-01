import ast
from pathlib import Path
from typing import List
from langchain_core.documents import Document


class CodeParser(ast.NodeVisitor):
    """An AST visitor to extract functions, classes, and their docstrings."""
    
    def __init__(self, file_path: str, source: str, repo_path: Path):
        self.file_path = file_path
        self.source = source
        self.documents = []
        self.repo_path = repo_path

    def visit_FunctionDef(self, node: ast.FunctionDef):
        docstring = ast.get_docstring(node)
        if docstring:
            # Reconstruct a simple signature
            signature = f"def {node.name}({ast.unparse(node.args)}):"
            content = f"{signature}\n\n{docstring}"
            self.documents.append(Document(
                page_content=content,
                metadata={"source": f"{self.source}-api", "file": self.file_path, "object": node.name}
            ))
        self.generic_visit(node)  # Continue visiting children

    def visit_ClassDef(self, node: ast.ClassDef):
        docstring = ast.get_docstring(node)
        if docstring:
            signature = f"class {node.name}:"
            content = f"{signature}\n\n{docstring}"
            self.documents.append(Document(
                page_content=content,
                metadata={"source": f"{self.source}-api", "file": self.file_path, "object": node.name}
            ))
        self.generic_visit(node)  # Continue visiting children


class DocumentParser:
    """Handles parsing of source code and documentation files."""
    
    def __init__(self, repo_paths: dict):
        self.repo_paths = repo_paths
    
    def parse_source_code(self, src_path: Path, source: str, repo_path: Path) -> List[Document]:
        """Parses Python source files to extract API documentation."""
        print(f"Parsing source code in: {src_path}...")
        documents = []
        for py_file in src_path.rglob("*.py"):
            try:
                file_content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(file_content)
                relative_path = str(py_file.relative_to(repo_path))
                parser = CodeParser(relative_path, source, repo_path)
                parser.visit(tree)
                documents.extend(parser.documents)
            except Exception as e:
                print(f"--> Could not parse {py_file}: {e}")
                
        print(f"-> Found {len(documents)} docstrings in source code.")
        return documents

    def parse_guides(self, docs_path: Path, source: str, repo_path: Path) -> List[Document]:
        """Parses Markdown and RST guides, loading each file as a single document."""
        print(f"Parsing guides in: {docs_path}...")
        documents = []
        
        # Parse Markdown files
        for md_file in docs_path.rglob("*.md"):
            file_content = md_file.read_text(encoding="utf-8")
            relative_path = str(md_file.relative_to(repo_path))
            doc = Document(
                page_content=file_content,
                metadata={"source": f"{source}-guide", "file": relative_path}
            )
            documents.append(doc)
        
        # Parse RST files
        for rst_file in docs_path.rglob("*.rst"):
            file_content = rst_file.read_text(encoding="utf-8")
            relative_path = str(rst_file.relative_to(repo_path))
            doc = Document(
                page_content=file_content,
                metadata={"source": f"{source}-guide", "file": relative_path}
            )
            documents.append(doc)
                
        print(f"-> Found {len(documents)} guide files.")
        return documents
    
    def parse_all_repositories(self) -> List[Document]:
        """Parse all configured repositories and return combined documents."""
        all_documents = []
        
        for repo_name, repo_path in self.repo_paths.items():
            if not repo_path.exists():
                print(f"Warning: Repository path {repo_path} does not exist. Skipping {repo_name}.")
                continue
                
            docs_path = repo_path / "docs"
            src_path = repo_path / (repo_name.replace("-", "_") if repo_name != "sleap" else "sleap")
            
            # Parse guides
            if docs_path.exists():
                guide_docs = self.parse_guides(docs_path, repo_name, repo_path)
                all_documents.extend(guide_docs)
            else:
                print(f"Warning: Docs path {docs_path} does not exist for {repo_name}")
            
            # Parse source code
            if src_path.exists():
                api_docs = self.parse_source_code(src_path, repo_name, repo_path)
                all_documents.extend(api_docs)
            else:
                print(f"Warning: Source path {src_path} does not exist for {repo_name}")
        
        print(f"\n✅ Total documents collected: {len(all_documents)}")
        return all_documents
