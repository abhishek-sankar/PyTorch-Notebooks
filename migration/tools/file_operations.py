"""
File Operations Tool for Java Migration System

Provides safe file system operations with backup and rollback capabilities.
Used by agents for reading, writing, and modifying source code files.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)


class FileOperations:
    """
    Safe file operations with backup and rollback capabilities.
    
    Provides agents with the ability to read, write, and modify files
    while maintaining backups for rollback scenarios.
    """
    
    def __init__(self, enable_backup: bool = True):
        self.enable_backup = enable_backup
        self.backup_registry: Dict[str, str] = {}  # file_path -> backup_path
        self.operation_log: List[Dict[str, Any]] = []
        
    def read_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read a file and return its contents.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Dictionary with file contents and metadata
        """
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}",
                    "content": None
                }
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file metadata
            stat = file_path.stat()
            
            result = {
                "success": True,
                "content": content,
                "file_path": str(file_path),
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "encoding": "utf-8",
                "line_count": len(content.splitlines()) if content else 0
            }
            
            logger.info(f"Successfully read file: {file_path}")
            return result
            
        except UnicodeDecodeError:
            # Try reading as binary for non-text files
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                return {
                    "success": True,
                    "content": content,
                    "file_path": str(file_path),
                    "size": len(content),
                    "encoding": "binary",
                    "is_binary": True
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read file as binary: {e}",
                    "content": None
                }
                
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def write_file(self, file_path: Union[str, Path], content: str, create_backup: bool = None) -> Dict[str, Any]:
        """
        Write content to a file with optional backup.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            create_backup: Whether to create backup (uses default if None)
            
        Returns:
            Dictionary with operation result
        """
        file_path = Path(file_path)
        create_backup = create_backup if create_backup is not None else self.enable_backup
        
        try:
            # Create backup if file exists and backup is enabled
            backup_path = None
            if create_backup and file_path.exists():
                backup_path = self._create_backup(file_path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log the operation
            operation = {
                "type": "write",
                "file_path": str(file_path),
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat(),
                "content_size": len(content),
                "line_count": len(content.splitlines())
            }
            self.operation_log.append(operation)
            
            logger.info(f"Successfully wrote file: {file_path}")
            return {
                "success": True,
                "file_path": str(file_path),
                "backup_path": backup_path,
                "bytes_written": len(content.encode('utf-8')),
                "lines_written": len(content.splitlines())
            }
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": str(file_path)
            }
    
    def modify_file(self, file_path: Union[str, Path], modifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply multiple modifications to a file.
        
        Args:
            file_path: Path to the file to modify
            modifications: List of modification operations
                Each modification should have:
                - type: "replace", "insert", "delete"
                - params: operation-specific parameters
                
        Returns:
            Dictionary with operation result
        """
        file_path = Path(file_path)
        
        try:
            # Read current content
            read_result = self.read_file(file_path)
            if not read_result["success"]:
                return read_result
            
            content = read_result["content"]
            original_content = content
            
            # Apply modifications
            applied_modifications = []
            
            for i, mod in enumerate(modifications):
                try:
                    if mod["type"] == "replace":
                        content = self._apply_replace_modification(content, mod["params"])
                    elif mod["type"] == "insert":
                        content = self._apply_insert_modification(content, mod["params"])
                    elif mod["type"] == "delete":
                        content = self._apply_delete_modification(content, mod["params"])
                    else:
                        logger.warning(f"Unknown modification type: {mod['type']}")
                        continue
                    
                    applied_modifications.append(mod)
                    
                except Exception as e:
                    logger.error(f"Failed to apply modification {i}: {e}")
                    continue
            
            # Write modified content if changes were made
            if content != original_content:
                write_result = self.write_file(file_path, content)
                
                if write_result["success"]:
                    return {
                        "success": True,
                        "file_path": str(file_path),
                        "modifications_applied": len(applied_modifications),
                        "total_modifications": len(modifications),
                        "backup_path": write_result.get("backup_path"),
                        "content_changed": True
                    }
                else:
                    return write_result
            else:
                return {
                    "success": True,
                    "file_path": str(file_path),
                    "modifications_applied": 0,
                    "content_changed": False,
                    "message": "No changes needed"
                }
            
        except Exception as e:
            logger.error(f"Failed to modify file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": str(file_path)
            }
    
    def find_and_replace(self, file_path: Union[str, Path], find_replace_pairs: List[tuple]) -> Dict[str, Any]:
        """
        Find and replace text patterns in a file.
        
        Args:
            file_path: Path to the file
            find_replace_pairs: List of (find, replace) tuples
            
        Returns:
            Dictionary with operation result
        """
        modifications = []
        
        for find_text, replace_text in find_replace_pairs:
            modifications.append({
                "type": "replace",
                "params": {
                    "find": find_text,
                    "replace": replace_text
                }
            })
        
        return self.modify_file(file_path, modifications)
    
    def backup_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Dictionary with backup result
        """
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}"
                }
            
            backup_path = self._create_backup(file_path)
            
            return {
                "success": True,
                "original_path": str(file_path),
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_path": str(file_path)
            }
    
    def restore_from_backup(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Restore a file from its backup.
        
        Args:
            file_path: Path to the file to restore
            
        Returns:
            Dictionary with restore result
        """
        file_path_str = str(file_path)
        
        try:
            if file_path_str not in self.backup_registry:
                return {
                    "success": False,
                    "error": f"No backup found for file: {file_path}",
                    "file_path": file_path_str
                }
            
            backup_path = self.backup_registry[file_path_str]
            
            if not Path(backup_path).exists():
                return {
                    "success": False,
                    "error": f"Backup file does not exist: {backup_path}",
                    "file_path": file_path_str
                }
            
            # Restore the file
            shutil.copy2(backup_path, file_path)
            
            logger.info(f"Successfully restored file from backup: {file_path}")
            return {
                "success": True,
                "file_path": file_path_str,
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to restore file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path_str
            }
    
    def list_files(self, directory: Union[str, Path], pattern: str = "*", recursive: bool = True) -> Dict[str, Any]:
        """
        List files in a directory matching a pattern.
        
        Args:
            directory: Directory to search in
            pattern: File pattern (e.g., "*.java")
            recursive: Whether to search recursively
            
        Returns:
            Dictionary with file list
        """
        directory = Path(directory)
        
        try:
            if not directory.exists():
                return {
                    "success": False,
                    "error": f"Directory does not exist: {directory}",
                    "files": []
                }
            
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            # Filter only files (not directories)
            files = [f for f in files if f.is_file()]
            
            file_info = []
            for file_path in files:
                stat = file_path.stat()
                file_info.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": file_path.suffix
                })
            
            return {
                "success": True,
                "directory": str(directory),
                "pattern": pattern,
                "file_count": len(file_info),
                "files": file_info
            }
            
        except Exception as e:
            logger.error(f"Failed to list files in {directory}: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": str(directory),
                "files": []
            }
    
    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the log of all file operations performed"""
        return self.operation_log.copy()
    
    def get_backup_registry(self) -> Dict[str, str]:
        """Get the registry of all backups created"""
        return self.backup_registry.copy()
    
    def cleanup_backups(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up old backup files.
        
        Args:
            max_age_hours: Maximum age of backups to keep
            
        Returns:
            Dictionary with cleanup result
        """
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            for original_path, backup_path in list(self.backup_registry.items()):
                backup_file = Path(backup_path)
                
                if backup_file.exists():
                    # Check backup age
                    backup_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    age_hours = (current_time - backup_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        backup_file.unlink()
                        del self.backup_registry[original_path]
                        cleaned_count += 1
                        logger.info(f"Cleaned up old backup: {backup_path}")
                else:
                    # Remove registry entry for non-existent backup
                    del self.backup_registry[original_path]
                    cleaned_count += 1
            
            return {
                "success": True,
                "cleaned_count": cleaned_count,
                "remaining_backups": len(self.backup_registry)
            }
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Private helper methods
    
    def _create_backup(self, file_path: Path) -> str:
        """Create a backup of the specified file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}_backup{file_path.suffix}"
        backup_path = file_path.parent / ".backups" / backup_name
        
        # Create backup directory if it doesn't exist
        backup_path.parent.mkdir(exist_ok=True)
        
        # Copy the file
        shutil.copy2(file_path, backup_path)
        
        # Register the backup
        self.backup_registry[str(file_path)] = str(backup_path)
        
        logger.info(f"Created backup: {file_path} -> {backup_path}")
        return str(backup_path)
    
    def _apply_replace_modification(self, content: str, params: Dict[str, Any]) -> str:
        """Apply a replace modification"""
        find_text = params["find"]
        replace_text = params["replace"]
        
        if params.get("regex", False):
            import re
            content = re.sub(find_text, replace_text, content)
        else:
            content = content.replace(find_text, replace_text)
        
        return content
    
    def _apply_insert_modification(self, content: str, params: Dict[str, Any]) -> str:
        """Apply an insert modification"""
        lines = content.splitlines(keepends=True)
        line_number = params.get("line", 0)  # 0-based
        text_to_insert = params["text"]
        
        if line_number < 0:
            line_number = len(lines) + line_number + 1
        
        if 0 <= line_number <= len(lines):
            lines.insert(line_number, text_to_insert + "\n")
        
        return "".join(lines)
    
    def _apply_delete_modification(self, content: str, params: Dict[str, Any]) -> str:
        """Apply a delete modification"""
        if "line" in params:
            # Delete specific line
            lines = content.splitlines(keepends=True)
            line_number = params["line"]  # 0-based
            
            if 0 <= line_number < len(lines):
                del lines[line_number]
            
            return "".join(lines)
        elif "text" in params:
            # Delete specific text
            text_to_delete = params["text"]
            return content.replace(text_to_delete, "")
        else:
            return content