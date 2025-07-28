"""
LangChain tools for file operations in migration agents
"""
from pathlib import Path
from typing import List
from langchain_core.tools import tool
import re

@tool
def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        Path(file_path).write_text(content, encoding='utf-8')
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def find_replace(file_path: str, find_text: str, replace_text: str) -> str:
    """Find and replace text in a file."""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        count = content.count(find_text)
        if count > 0:
            new_content = content.replace(find_text, replace_text)
            Path(file_path).write_text(new_content, encoding='utf-8')
            return f"Made {count} replacements in {file_path}"
        else:
            return f"No occurrences of '{find_text}' found in {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_java_files(directory: str) -> str:
    """List all Java files in a directory and subdirectories."""
    try:
        java_files = [str(f) for f in Path(directory).rglob("*.java")]
        return f"Found {len(java_files)} Java files:\n" + "\n".join(java_files)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_files(directory: str, pattern: str) -> str:
    """Search for a regex pattern in Java files."""
    try:
        matches = []
        java_files = [str(f) for f in Path(directory).rglob("*.java")]
        
        for file_path in java_files:
            content = Path(file_path).read_text(encoding='utf-8')
            for line_num, line in enumerate(content.split('\n'), 1):
                if re.search(pattern, line):
                    matches.append(f"{file_path}:{line_num}: {line.strip()}")
        
        if matches:
            return f"Found {len(matches)} matches:\n" + "\n".join(matches[:20])
        else:
            return f"No matches found for pattern '{pattern}'"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def file_exists(file_path: str) -> str:
    """Check if a file exists."""
    return str(Path(file_path).exists())

# Collect all file tools
file_tools = [read_file, write_file, find_replace, list_java_files, search_files, file_exists]