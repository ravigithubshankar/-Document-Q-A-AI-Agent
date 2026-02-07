"""
Document Processing Module for Enterprise Document Q&A Agent
Handles PDF extraction, text processing, and document management with security standards
"""

import os
import re
import tempfile
import warnings
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class ProcessingConfig:
    """Configuration for document processing"""
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200
    MAX_FILE_SIZE_MB = 50  # Security: Limit file size
    
    # Security: Allowed file extensions
    ALLOWED_EXTENSIONS = {'.pdf'}
    
    # Security: Malicious pattern detection
    MALICIOUS_PATTERNS = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'on\w+=\".*?\"',
        r'<\?php.*?\?>',
        r'<\?.*?\?>',
        r'<!DOCTYPE.*?>',
    ]

# ==================== SECURITY VALIDATOR ====================
class SecurityValidator:
    """Security validation and sanitization utilities"""
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security"""
        if not filename:
            return False
        
        # Check file extension
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ProcessingConfig.ALLOWED_EXTENSIONS:
            return False
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for suspicious characters
        suspicious_chars = [';', '|', '&', '`', '$', '(']
        if any(char in filename for char in suspicious_chars):
            return False
        
        return True
    
    @staticmethod
    def validate_file_size(file_bytes: bytes) -> bool:
        """Validate file size for security"""
        max_size = ProcessingConfig.MAX_FILE_SIZE_MB * 1024 * 1024
        return len(file_bytes) <= max_size
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove potentially malicious content from text"""
        if not text:
            return ""
        
        for pattern in ProcessingConfig.MALICIOUS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        return text.strip()
    
    @staticmethod
    def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize document metadata"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                sanitized[key] = SecurityValidator.sanitize_text(str(value)[:500])  # Limit length
            else:
                sanitized[key] = str(value)[:500] if value else ""
        return sanitized

# ==================== DOCUMENT EXTRACTOR ====================
class DocumentExtractor:
    """Handles PDF text extraction with multiple fallback methods"""
    
    @staticmethod
    def extract_with_pdfplumber(file_path: str) -> Optional[Dict[str, Any]]:
        """Extract text using pdfplumber (primary method)"""
        try:
            with pdfplumber.open(file_path) as pdf:
                all_text = ""
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    all_text += f"\n--- Page {page_num + 1} ---\n{text}"
                    
                    # Extract tables
                    try:
                        page_tables = page.extract_tables()
                        for table in page_tables:
                            if table and len(table) > 0:
                                tables.append({
                                    "page": page_num + 1,
                                    "data": table[:10]  # Limit rows for security
                                })
                    except:
                        pass
                
                # Extract metadata safely
                metadata = {}
                try:
                    if hasattr(pdf, 'metadata') and pdf.metadata:
                        metadata = {
                            "title": str(pdf.metadata.get("Title", "")),
                            "author": str(pdf.metadata.get("Author", "")),
                            "subject": str(pdf.metadata.get("Subject", ""))
                        }
                except:
                    pass
                
                return {
                    "num_pages": len(pdf.pages),
                    "text": all_text,
                    "tables": tables,
                    "metadata": metadata,
                    "method": "pdfplumber"
                }
                
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {str(e)[:100]}")
            return None
    
    @staticmethod
    def extract_with_pypdf2(file_path: str) -> Optional[Dict[str, Any]]:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                all_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text() or ""
                    all_text += f"\n--- Page {page_num + 1} ---\n{text}"
                
                # Extract metadata safely
                metadata = {}
                try:
                    if pdf_reader.metadata:
                        metadata = {
                            "title": str(pdf_reader.metadata.get("/Title", "")),
                            "author": str(pdf_reader.metadata.get("/Author", "")),
                            "subject": str(pdf_reader.metadata.get("/Subject", ""))
                        }
                except:
                    pass
                
                return {
                    "num_pages": len(pdf_reader.pages),
                    "text": all_text,
                    "tables": [],
                    "metadata": metadata,
                    "method": "pypdf2"
                }
                
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)[:100]}")
            return None
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> Optional[Dict[str, Any]]:
        """Main extraction method with fallback strategies and security"""
        try:
            # Try pdfplumber first
            result = DocumentExtractor.extract_with_pdfplumber(file_path)
            
            # Fallback to PyPDF2 if needed
            if not result or len(result.get("text", "")) < 100:
                result = DocumentExtractor.extract_with_pypdf2(file_path)
            
            if result:
                # Security: Sanitize extracted content
                result["text"] = SecurityValidator.sanitize_text(result["text"])
                result["metadata"] = SecurityValidator.sanitize_metadata(result.get("metadata", {}))
                
                # Security: Limit table data
                for table in result.get("tables", []):
                    if "data" in table:
                        table["data"] = table["data"][:5]  # Limit to 5 rows
            
            return result
            
        except Exception as e:
            st.error(f"Document extraction failed: {str(e)[:200]}")
            return None

# ==================== DOCUMENT PROCESSOR ====================
class DocumentProcessor:
    """Main document processing class with enterprise features"""
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.security = SecurityValidator()
    
    def process_single_file(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Process a single PDF file with security checks"""
        try:
            # Security: Validate filename
            if not self.security.validate_filename(filename):
                st.error(f"Security: Invalid filename '{filename}'")
                return None
            
            # Security: Validate file size
            if not self.security.validate_file_size(file_bytes):
                st.error(f"Security: File too large (max {ProcessingConfig.MAX_FILE_SIZE_MB}MB)")
                return None
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Extract content
                extraction_result = DocumentExtractor.extract_from_pdf(tmp_path)
                
                if extraction_result:
                    # Create document info
                    doc_info = {
                        "filename": filename,
                        "num_pages": extraction_result.get("num_pages", 0),
                        "metadata": extraction_result.get("metadata", {}),
                        "tables": extraction_result.get("tables", []),
                        "full_text": extraction_result.get("text", ""),
                        "extraction_method": extraction_result.get("method", "unknown"),
                        "processing_time": time.time()
                    }
                    
                    # Create chunks
                    doc_info["chunks"] = self._create_chunks(doc_info["full_text"])
                    
                    # Security: Add processing signature
                    doc_info["processed_by"] = "EnterpriseDocQA"
                    doc_info["security_level"] = "sanitized"
                    
                    return doc_info
                else:
                    st.warning(f"Failed to extract content from {filename}")
                    return None
                    
            finally:
                # Security: Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)[:200]}")
            return None
    
    def process_multiple_files(self, uploaded_files: List) -> List[Dict[str, Any]]:
        """Process multiple uploaded files with progress tracking"""
        processed_docs = []
        
        if not uploaded_files:
            return processed_docs
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"🔒 Processing: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
            
            result = self.process_single_file(uploaded_file.getvalue(), uploaded_file.name)
            
            if result:
                processed_docs.append(result)
                st.success(f"✅ {uploaded_file.name}: {len(result.get('chunks', []))} chunks")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        # Update documents and statistics
        self.documents = processed_docs
        
        return processed_docs
    
    def _create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into manageable chunks with section detection"""
        if not text:
            return []
        
        # Security: Limit text length before chunking
        text = text[:1000000]  # 1MB limit
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # Security: Skip suspiciously large paragraphs
            if para_size > 10000:
                continue
            
            if current_size + para_size > ProcessingConfig.CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "size": len(chunk_text),
                    "section": self._detect_section(chunk_text),
                    "security_hash": hash(chunk_text[:100])  # For integrity check
                })
                
                # Keep overlap for context
                if current_chunk:
                    overlap_size = min(len(current_chunk), 
                                     ProcessingConfig.CHUNK_OVERLAP // 50)
                    current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                    current_size = sum(len(p) for p in current_chunk)
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "size": len(chunk_text),
                "section": self._detect_section(chunk_text),
                "security_hash": hash(chunk_text[:100])
            })
        
        return chunks
    
    def _detect_section(self, text: str) -> str:
        """Detect document section from text content"""
        text_lower = text.lower()[:500]  # Limit for performance
        
        section_patterns = {
            "Abstract": [r'\babstract\b', r'\bsummary\b'],
            "Introduction": [r'\bintroduction\b', r'\bbackground\b'],
            "Methodology": [r'\bmethod\b', r'\bmethodology\b', r'\bapproach\b', r'\bexperiment\b'],
            "Results": [r'\bresult\b', r'\bfinding\b', r'\bdata\b', r'\btable\b', r'\bfigure\b'],
            "Discussion": [r'\bdiscussion\b', r'\banalysis\b', r'\binterpretation\b'],
            "Conclusion": [r'\bconclusion\b', r'\bconcluding\b'],
            "References": [r'\breference\b', r'\bbibliography\b']
        }
        
        for section, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section
        
        return "General"
    
    def search_in_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search for query in processed documents with relevance scoring"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Security: Sanitize search query
        query_lower = SecurityValidator.sanitize_text(query_lower)
        
        for doc in self.documents:
            doc_results = []
            
            for chunk_idx, chunk in enumerate(doc.get("chunks", [])):
                chunk_text = chunk.get("text", "").lower()
                
                # Calculate relevance score
                relevance = 0
                
                # Exact phrase match (boosted)
                if query_lower in chunk_text:
                    relevance += 10
                
                # Individual word matches
                word_matches = sum(1 for word in query_words if word in chunk_text)
                relevance += word_matches * 2
                
                # Boost for important sections
                if chunk.get("section") in ["Abstract", "Conclusion", "Results"]:
                    relevance += 3
                
                if relevance > 0:
                    doc_results.append({
                        "document": doc.get("filename", "Unknown"),
                        "section": chunk.get("section", "General"),
                        "text": self._truncate_text(chunk.get("text", ""), 300),
                        "relevance_score": min(relevance, 20),  # Cap at 20
                        "chunk_index": chunk_idx,
                        "security_hash": chunk.get("security_hash")
                    })
            
            # Keep top results per document
            doc_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            results.extend(doc_results[:3])
        
        # Sort all results by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:10]  # Return top 10 results
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Safely truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.7:
            return truncated[:last_period + 1] + " [...]"
        
        return truncated + " [...]"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        if not self.documents:
            return {
                "total_documents": 0,
                "total_pages": 0,
                "total_words": 0,
                "total_chunks": 0,
                "total_tables": 0,
                "avg_chunk_size": 0,
                "sections_distribution": {},
                "extraction_methods": {},
                "security": {"all_sanitized": False, "processed_by": "EnterpriseDocQA"}
            }
        
        total_pages = 0
        total_words = 0
        total_chunks = 0
        total_tables = 0
        
        sections_count = {}
        extraction_methods = {}
        
        for doc in self.documents:
            total_pages += doc.get('num_pages', 0)
            # Count words properly
            full_text = doc.get('full_text', '')
            total_words += len(full_text.split())
            total_chunks += len(doc.get('chunks', []))
            total_tables += len(doc.get('tables', []))
            
            # Count extraction methods
            method = doc.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            # Count sections
            for chunk in doc.get('chunks', []):
                section = chunk.get('section', 'General')
                sections_count[section] = sections_count.get(section, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_pages": total_pages,
            "total_words": total_words,
            "total_chunks": total_chunks,
            "total_tables": total_tables,
            "avg_chunk_size": total_words / total_chunks if total_chunks > 0 else 0,
            "sections_distribution": sections_count,
            "extraction_methods": extraction_methods,
            "security": {
                "all_sanitized": all(d.get('security_level') == 'sanitized' for d in self.documents),
                "processed_by": "EnterpriseDocQA"
            }
        }
    
    def clear_documents(self) -> None:
        """Clear all processed documents"""
        self.documents = []
        st.success("All documents cleared successfully.")

# ==================== ERROR HANDLER ====================
class ProcessingErrorHandler:
    """Centralized error handling for document processing"""
    
    @staticmethod
    def handle_extraction_error(filename: str, error: Exception) -> str:
        """Handle extraction errors gracefully"""
        error_msg = str(error).lower()
        
        if "encrypted" in error_msg:
            return f"Security: Document '{filename}' is encrypted and cannot be processed."
        elif "corrupt" in error_msg or "damaged" in error_msg:
            return f"Error: Document '{filename}' appears to be corrupted."
        elif "permission" in error_msg:
            return f"Security: Permission denied while processing '{filename}'."
        elif "memory" in error_msg:
            return f"Error: Insufficient memory to process '{filename}'."
        else:
            return f"Error processing '{filename}': {str(error)[:100]}"
    
    @staticmethod
    def validate_processing_environment() -> bool:
        """Validate that required packages are installed"""
        missing_packages = []
        
        try:
            import PyPDF2
        except ImportError:
            missing_packages.append("PyPDF2")
        
        try:
            import pdfplumber
        except ImportError:
            missing_packages.append("pdfplumber")
        
        if missing_packages:
            st.error(f"Missing packages: {', '.join(missing_packages)}")
            st.info(f"Install with: pip install {' '.join(missing_packages)}")
            return False
        
        return True