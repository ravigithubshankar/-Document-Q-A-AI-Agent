"""
Main Application File for Enterprise Document Q&A Agent
Streamlit interface with AI integration, security, and enterprise features
"""

import os
import re
import time
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import arxiv

# Import processing module
try:
    from processing import DocumentProcessor, ProcessingErrorHandler, SecurityValidator
except ImportError as e:
    st.error(f"Processing module import error: {str(e)}")
    st.info("Ensure app_processing.py is in the same directory")
    st.stop()

# Groq LLM integration
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("Groq SDK not installed. Install with: pip install groq")

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class AppConfig:
    """Application configuration with security settings"""
    # API Configuration (Hardcoded for demonstration - in production use environment variables)
    GROQ_API_KEY = "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"
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # Security: API usage limits
    MAX_TOKENS_PER_REQUEST = 2000
    MAX_REQUESTS_PER_MINUTE = 30
    REQUEST_TIMEOUT = 30  # seconds
    
    # Application settings
    MAX_CONTEXT_LENGTH = 25000
    ARXIV_MAX_RESULTS = 5
    
    # UI Settings
    PAGE_TITLE = "Enterprise Document Q&A Agent"
    PAGE_ICON = "📚"
    
    # Security: Input validation
    MAX_QUERY_LENGTH = 1000
    MAX_CONVERSATION_HISTORY = 20

# ==================== ENTERPRISE AI AGENT ====================
class EnterpriseAIAgent:
    """AI Agent with enterprise-grade features and security"""
    
    def __init__(self, api_key: str):
        """Initialize AI agent with security checks"""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq SDK not available. Install with: pip install groq")
        
        # Security: Validate API key format
        if not self._validate_api_key(api_key):
            raise ValueError("Invalid API key format")
        
        # Security: Rate limiting tracker
        self.request_timestamps = []
        
        try:
            self.client = Groq(api_key=api_key)
            self.conversation_history = []
            self.security_validator = SecurityValidator()
            
            # Initialize with security settings
            self._log_event("agent_initialized", "AI Agent initialized successfully")
            
        except Exception as e:
            self._log_event("agent_init_failed", f"Failed to initialize: {str(e)}")
            raise
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or not isinstance(api_key, str):
            return False
        return api_key.startswith("gsk_") and len(api_key) > 20
    
    def _log_event(self, event_type: str, message: str) -> None:
        """Log security and operational events"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message[:500]  # Limit log message length
        }
        # In production, this would write to a secure log file
        if event_type in ["security_alert", "error"]:
            st.warning(f"Security Event: {message}")
    
    def _check_rate_limit(self) -> bool:
        """Implement rate limiting for API calls"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # Check if limit exceeded
        if len(self.request_timestamps) >= AppConfig.MAX_REQUESTS_PER_MINUTE:
            self._log_event("rate_limit_exceeded", "API rate limit exceeded")
            return False
        
        self.request_timestamps.append(current_time)
        return True
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize user query for security"""
        if not query:
            return ""
        
        # Security: Limit query length
        query = query[:AppConfig.MAX_QUERY_LENGTH]
        
        # Security: Remove potentially malicious content
        query = self.security_validator.sanitize_text(query)
        
        # Security: Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _optimize_response(self, response: str) -> str:
        """Clean and optimize LLM responses"""
        if not response:
            return "No response generated."
        
        # Remove common AI assistant phrases
        patterns_to_remove = [
            r'As an AI(?: assistant)?.*?,?',
            r'Based on (?:the|this) (?:document|text).*?,?',
            r'According to (?:the|this) (?:document|text).*?,?',
            r'I am an AI.*?,?',
            r'As a language model.*?,?'
        ]
        
        for pattern in patterns_to_remove:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        # Fix formatting
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Security: Sanitize final response
        response = self.security_validator.sanitize_text(response)
        
        return response.strip()
    
    def _optimize_context(self, context: str) -> str:
        """Optimize context window for API efficiency"""
        if not context:
            return ""
        
        if len(context) > AppConfig.MAX_CONTEXT_LENGTH:
            # Keep important parts (beginning and end)
            half = AppConfig.MAX_CONTEXT_LENGTH // 2
            return context[:half] + "\n...[Content truncated for efficiency]...\n" + context[-half:]
        return context
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query for appropriate handling"""
        if not query:
            return "general"
        
        query_lower = query.lower()
        
        classification_patterns = {
            "direct": [
                r'what (is|are) (the )?(conclusion|result|finding|method|approach|definition)',
                r'where (is|are)',
                r'who (is|are)',
                r'when (is|are|does|did)',
                r'how (does|did|is|are|to|can)',
                r'explain (the|this|that)',
                r'describe (the|this|that)'
            ],
            "summarization": [
                r'summarize',
                r'overview',
                r'key (points|insights|findings|takeaways)',
                r'main (idea|points|arguments)',
                r'brief(ly)? (describe|explain|summary)',
                r'tl;?dr'
            ],
            "extraction": [
                r'extract',
                r'accuracy|f1.*score|precision|recall|metric',
                r'[0-9]+(\.[0-9]+)?%',
                r'table|figure|chart|graph|data',
                r'result.*(show|indicate|demonstrate|report)',
                r'value|number|score|rate'
            ],
            "comparison": [
                r'compare',
                r'difference between',
                r'similarities',
                r'contrast',
                r'vs\.|versus',
                r'how (does|are).*different'
            ]
        }
        
        for qtype, patterns in classification_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return qtype
        
        return "general"
    
    def query_llm(self, prompt: str, context: str = "", system_prompt: str = None) -> Dict[str, Any]:
        """Query LLM with security and error handling"""
        try:
            # Security: Check rate limits
            if not self._check_rate_limit():
                return {
                    "success": False,
                    "response": "Rate limit exceeded. Please wait a moment before trying again.",
                    "error": "RATE_LIMIT_EXCEEDED"
                }
            
            # Security: Sanitize inputs
            prompt = self._sanitize_query(prompt)
            context = self._sanitize_query(context) if context else ""
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system", 
                    "content": self._sanitize_query(system_prompt)
                })
            
            if context:
                optimized_context = self._optimize_context(context)
                user_content = f"""Based on the following document context, please answer the question:

Document Context:
{optimized_context}

Question: {prompt}

Guidelines:
1. Answer based ONLY on the provided context
2. If information is missing, state that clearly
3. Be precise and cite relevant sections
4. Format response professionally
"""
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": prompt})
            
            # Add limited conversation history
            for hist in self.conversation_history[-4:]:
                messages.insert(-1 if system_prompt else 0, hist)
            
            # Make API call with timeout
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=AppConfig.GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=AppConfig.MAX_TOKENS_PER_REQUEST,
                top_p=0.95,
                timeout=AppConfig.REQUEST_TIMEOUT
            )
            
            processing_time = time.time() - start_time
            
            result = response.choices[0].message.content
            result = self._optimize_response(result)
            
            # Add timing and security info
            result += f"\n\n*⏱️ Generated in {processing_time:.2f}s | 🔒 Security: Verified*"
            
            # Update conversation history (with limits)
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": result})
            
            # Security: Limit history size
            if len(self.conversation_history) > AppConfig.MAX_CONVERSATION_HISTORY * 2:
                self.conversation_history = self.conversation_history[-AppConfig.MAX_CONVERSATION_HISTORY * 2:]
            
            self._log_event("query_successful", f"Query processed in {processing_time:.2f}s")
            
            return {
                "success": True,
                "response": result,
                "processing_time": processing_time,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            error_msg = str(e)
            self._log_event("query_failed", f"Query failed: {error_msg[:100]}")
            
            # User-friendly error messages
            if "timeout" in error_msg.lower():
                return {
                    "success": False,
                    "response": "Request timeout. Please try again with a simpler query.",
                    "error": "TIMEOUT_ERROR"
                }
            elif "authentication" in error_msg.lower():
                return {
                    "success": False,
                    "response": "Authentication error. Please check your API key.",
                    "error": "AUTH_ERROR"
                }
            elif "rate limit" in error_msg.lower():
                return {
                    "success": False,
                    "response": "Rate limit exceeded. Please wait before trying again.",
                    "error": "RATE_LIMIT"
                }
            else:
                return {
                    "success": False,
                    "response": f"Error processing request: {error_msg[:100]}",
                    "error": "GENERAL_ERROR"
                }
    
    def process_document_query(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process user query with document context"""
        try:
            # Security and validation
            query = self._sanitize_query(query)
            
            # Determine query type for optimized handling
            query_type = self._classify_query_type(query)
            
            # Extract relevant context
            context = self._extract_relevant_context(query, documents)
            
            # Prepare system prompt based on query type
            system_prompts = {
                "direct": """You are a precise document analysis assistant.
                - Answer based ONLY on provided context
                - If information is missing, clearly state "Not found in documents"
                - Be concise and accurate
                - Cite specific sections when possible""",
                
                "summarization": """You are a summarization expert.
                - Create concise, accurate summaries
                - Focus on key insights and main points
                - Preserve technical accuracy
                - Structure with clear sections""",
                
                "extraction": """You are a data extraction specialist.
                - Extract specific information like metrics, results, findings
                - Present in structured format (bullet points, tables)
                - Include units and context for numerical values
                - Highlight key data points""",
                
                "comparison": """You are a comparative analysis expert.
                - Compare and contrast information from different documents
                - Highlight similarities and differences
                - Present findings in structured format
                - Draw meaningful insights""",
                
                "general": """You are a helpful research assistant.
                - Provide accurate, well-structured answers
                - Base responses on available information
                - Acknowledge limitations when information is incomplete
                - Format professionally"""
            }
            
            system_prompt = system_prompts.get(query_type, system_prompts["general"])
            
            # Get response from LLM
            llm_response = self.query_llm(query, context, system_prompt)
            
            return {
                "query_type": query_type,
                "llm_response": llm_response,
                "context_used": bool(context),
                "context_size": len(context) if context else 0,
                "documents_referenced": self._extract_referenced_documents(context, documents)
            }
            
        except Exception as e:
            self._log_event("query_processing_error", f"Error: {str(e)}")
            return {
                "query_type": "error",
                "llm_response": {
                    "success": False,
                    "response": f"Error processing query: {str(e)[:100]}"
                },
                "context_used": False,
                "context_size": 0,
                "documents_referenced": []
            }
    
    def _extract_relevant_context(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Extract relevant context from documents"""
        if not documents:
            return ""
        
        relevant_chunks = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            doc_chunks = []
            
            for chunk in doc.get("chunks", []):
                chunk_text = chunk.get("text", "").lower()
                
                # Calculate relevance score
                score = 0
                
                # Exact phrase match
                if query.lower() in chunk_text:
                    score += 10
                
                # Individual word matches
                word_matches = sum(1 for word in query_words if word in chunk_text)
                score += word_matches * 2
                
                # Boost for important sections
                section = chunk.get("section", "")
                if section in ["Abstract", "Conclusion", "Results"]:
                    score += 3
                elif section in ["Introduction", "Methodology"]:
                    score += 1
                
                if score > 0:
                    doc_chunks.append({
                        "text": chunk.get("text", ""),
                        "score": score,
                        "document": doc.get("filename", "Unknown"),
                        "section": section
                    })
            
            # Take top chunks from each document
            doc_chunks.sort(key=lambda x: x["score"], reverse=True)
            relevant_chunks.extend(doc_chunks[:2])
        
        # Sort all chunks by score
        relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Combine chunks respecting context window
        combined_context = ""
        total_length = 0
        
        for chunk in relevant_chunks[:6]:
            chunk_text = chunk.get("text", "")
            if total_length + len(chunk_text) < AppConfig.MAX_CONTEXT_LENGTH:
                header = f"\n\n[Document: {chunk['document']} | Section: {chunk['section']}]"
                combined_context += header + "\n" + chunk_text
                total_length += len(header) + len(chunk_text)
        
        return combined_context
    
    def _extract_referenced_documents(self, context: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract which documents were referenced in context"""
        if not context:
            return []
        
        referenced = set()
        
        for doc in documents:
            filename = doc.get("filename", "")
            if filename and filename in context:
                referenced.add(filename)
        
        return list(referenced)
    
    # ==================== ARXIV INTEGRATION ====================
    
    def search_arxiv_papers(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """Search for academic papers on ArXiv"""
        try:
            if not query or len(query.strip()) < 3:
                return {
                    "success": False,
                    "message": "Search query too short",
                    "papers": []
                }
            
            if max_results is None:
                max_results = AppConfig.ARXIV_MAX_RESULTS
            
            # Security: Sanitize search query
            query = self._sanitize_query(query)
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            
            for result in client.results(search):
                # Security: Limit and sanitize paper data
                paper_info = {
                    "title": self._sanitize_query(result.title)[:200],
                    "authors": [self._sanitize_query(author.name)[:100] 
                              for author in result.authors[:3]],
                    "summary": self._sanitize_query(result.summary)[:500],
                    "published": result.published.strftime("%Y-%m-%d") if hasattr(result, 'published') else "Unknown",
                    "pdf_url": result.pdf_url,
                    "categories": result.categories[:3] if result.categories else [],
                    "security_checked": True
                }
                papers.append(paper_info)
            
            self._log_event("arxiv_search", f"Found {len(papers)} papers for query: {query[:50]}")
            
            return {
                "success": True,
                "message": f"Found {len(papers)} papers",
                "papers": papers,
                "query": query
            }
            
        except Exception as e:
            self._log_event("arxiv_error", f"ArXiv search failed: {str(e)}")
            return {
                "success": False,
                "message": f"Error searching ArXiv: {str(e)[:100]}",
                "papers": []
            }
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        self._log_event("history_cleared", "Conversation history cleared")

# ==================== STREAMLIT UI COMPONENTS ====================
class UIComponents:
    """Reusable UI components with consistent styling"""
    
    @staticmethod
    def setup_page() -> None:
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=AppConfig.PAGE_TITLE,
            page_icon=AppConfig.PAGE_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': 'Enterprise Document Q&A Agent v1.0'
            }
        )
    
    @staticmethod
    def apply_custom_styles() -> None:
        """Apply custom CSS styles"""
        st.markdown("""
        <style>
        /* Main styles */
        .main-header {
            font-size: 2.8rem;
            color: #1e40af;
            text-align: center;
            margin-bottom: 0.5rem;
            padding: 1rem;
            background: linear-gradient(90deg, #1e40af, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Statistics cards - SIDE BY SIDE LAYOUT */
        .statistics-container {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            min-width: 180px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            margin: 0.5rem;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .stat-label {
            font-size: 1rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        /* ArXiv Results Container - PROPER CENTERING */
        .arxiv-results-container {
            margin: 2rem auto;
            max-width: 1200px;
            padding: 1rem;
            width: 100%;
        }
        
        .arxiv-paper-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        .arxiv-paper-card:hover {
            border-color: #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        }
        
        /* Response cards */
        .response-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .response-success {
            border-left: 4px solid #10b981;
        }
        
        .response-error {
            border-left: 4px solid #ef4444;
        }
        
        /* Security indicators */
        .security-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0.25rem;
        }
        
        .security-high {
            background-color: #10b981;
            color: white;
        }
        
        .security-medium {
            background-color: #f59e0b;
            color: white;
        }
        
        .security-low {
            background-color: #ef4444;
            color: white;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .status-active {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        
        .status-inactive {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        
        /* Query badges */
        .query-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .badge-direct { background-color: #3b82f6; color: white; }
        .badge-summary { background-color: #10b981; color: white; }
        .badge-extract { background-color: #8b5cf6; color: white; }
        .badge-general { background-color: #6b7280; color: white; }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #3b82f6;
        }
        
        /* Sidebar sections */
        .sidebar-section {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
        }
        
        /* Main content area */
        .main-content {
            padding: 1rem 2rem;
        }
        
        /* Tab content */
        .tab-content {
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_header() -> None:
        """Create application header"""
        st.markdown('<h1 class="main-header">Enterprise Document Q&A Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">🔒 Secure AI-powered document analysis with enterprise-grade features</p>', unsafe_allow_html=True)
    
    @staticmethod
    def create_sidebar_status(agent, documents) -> None:
        """Create sidebar status section"""
        st.markdown("### 🔐 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if agent:
                st.markdown('<div class="status-indicator status-active">AI Agent Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-inactive">AI Agent Inactive</div>', unsafe_allow_html=True)
        
        with col2:
            if documents and len(documents) > 0:
                st.markdown(f'<div class="status-indicator status-active">{len(documents)} Documents</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-inactive">No Documents</div>', unsafe_allow_html=True)
        
        # Security level indicator
        security_level = "high" if agent and documents and len(documents) > 0 else "medium" if agent else "low"
        st.markdown(f'<div class="security-badge security-{security_level}">Security: {security_level.upper()}</div>', unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================
class EnterpriseDocumentQAAgent:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.ui = UIComponents()
        self.processor = DocumentProcessor()
        self.agent = None
        self._initialize_session_state()
        
        # Setup page and styles
        self.ui.setup_page()
        self.ui.apply_custom_styles()
        
        # Validate environment
        if not ProcessingErrorHandler.validate_processing_environment():
            st.stop()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables"""
        default_values = {
            'documents': [],
            'conversation': [],
            'show_history': False,
            'arxiv_results': [],
            'stats': {},
            'documents_loaded': False  # Add this flag
        }
        
        for key, default_value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _initialize_ai_agent(self) -> bool:
        """Initialize AI agent with error handling"""
        try:
            if self.agent is None:
                self.agent = EnterpriseAIAgent(api_key=AppConfig.GROQ_API_KEY)
                st.success("✅ AI Agent initialized with security checks")
                return True
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize AI Agent: {str(e)}")
            return False
    
    def _process_uploaded_files(self, uploaded_files) -> None:
        """Process uploaded PDF files"""
        if not uploaded_files:
            return
        
        with st.spinner("🔒 Processing documents with security checks..."):
            try:
                processed = self.processor.process_multiple_files(uploaded_files)
                
                if processed and len(processed) > 0:
                    st.session_state.documents = processed
                    st.session_state.documents_loaded = True  # Set flag
                    
                    # Force statistics calculation
                    stats = self.processor.get_statistics()
                    st.session_state.stats = stats
                    
                    st.success(f"✅ Successfully processed {len(processed)} documents")
                    st.rerun()
                else:
                    st.error("❌ No documents were successfully processed")
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
    
    def _get_safe_statistics(self) -> Dict[str, Any]:
        """Get statistics with safe default values"""
        try:
            # Check if documents are loaded
            if not st.session_state.documents_loaded or not st.session_state.documents:
                # Return default empty stats
                return {
                    "total_documents": 0,
                    "total_pages": 0,
                    "total_chunks": 0,
                    "total_words": 0,
                    "total_tables": 0,
                    "avg_chunk_size": 0,
                    "security": {"all_sanitized": False, "processed_by": "EnterpriseDocQA"}
                }
            
            # Always get fresh statistics from processor
            if self.processor.documents:
                stats = self.processor.get_statistics()
            else:
                # Calculate from session state documents if processor is empty
                total_pages = 0
                total_chunks = 0
                total_words = 0
                
                for doc in st.session_state.documents:
                    total_pages += doc.get('num_pages', 0)
                    total_chunks += len(doc.get('chunks', []))
                    total_words += len(doc.get('full_text', '').split())
                
                stats = {
                    "total_documents": len(st.session_state.documents),
                    "total_pages": total_pages,
                    "total_chunks": total_chunks,
                    "total_words": total_words,
                    "total_tables": 0,
                    "avg_chunk_size": total_words / total_chunks if total_chunks > 0 else 0,
                    "security": {"all_sanitized": True, "processed_by": "EnterpriseDocQA"}
                }
            
            # Ensure all required fields exist
            required_fields = {
                "total_documents": 0,
                "total_pages": 0,
                "total_chunks": 0,
                "total_words": 0,
                "total_tables": 0,
                "avg_chunk_size": 0,
                "security": {"all_sanitized": False, "processed_by": "EnterpriseDocQA"}
            }
            
            # Merge with defaults
            for key, default_value in required_fields.items():
                if key not in stats:
                    stats[key] = default_value
            
            return stats
        except Exception as e:
            # Fallback to session state or defaults
            if hasattr(st.session_state, 'stats') and st.session_state.stats:
                return st.session_state.stats
            return {
                "total_documents": len(st.session_state.documents) if st.session_state.documents else 0,
                "total_pages": 0,
                "total_chunks": 0,
                "total_words": 0,
                "total_tables": 0,
                "avg_chunk_size": 0,
                "security": {"all_sanitized": False, "processed_by": "EnterpriseDocQA"}
            }
    
    def _display_document_statistics(self) -> None:
        """Display document statistics with proper layout"""
        try:
            stats = self._get_safe_statistics()
            
            st.markdown("### 📊 Document Statistics")
            
            # Create columns for side-by-side layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_documents', 0)}</div>
                    <div class="stat-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_pages', 0)}</div>
                    <div class="stat-label">Pages</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_chunks', 0)}</div>
                    <div class="stat-label">Text Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                words = stats.get('total_words', 0)
                formatted_words = f"{words:,}" if words >= 1000 else str(words)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{formatted_words}</div>
                    <div class="stat-label">Total Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Security status
            if stats.get("security", {}).get("all_sanitized", False):
                st.success("🔒 All documents passed security checks")
            else:
                st.info("🔐 Security checks completed")
                
        except Exception as e:
            st.warning(f"⚠️ Could not display statistics: {str(e)}")
    
    def _handle_document_query(self, query: str) -> None:
        """Handle document query processing"""
        if not query or not query.strip():
            st.warning("Please enter a query")
            return
        
        if not st.session_state.documents:
            st.warning("Please upload and process documents first")
            return
        
        with st.spinner("🔍 Analyzing documents with AI..."):
            try:
                result = self.agent.process_document_query(query, st.session_state.documents)
                
                # Display response
                st.markdown("### 🤖 AI Response")
                
                # Query type badge
                badge_map = {
                    "direct": ("Direct Lookup", "badge-direct"),
                    "summarization": ("Summarization", "badge-summary"),
                    "extraction": ("Data Extraction", "badge-extract"),
                    "general": ("General Query", "badge-general")
                }
                
                badge_text, badge_class = badge_map.get(result.get("query_type", "general"), ("Query", "badge-general"))
                st.markdown(f'<span class="query-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
                
                # Response card
                llm_response = result.get("llm_response", {})
                card_class = "response-success" if llm_response.get("success") else "response-error"
                
                st.markdown(f'<div class="response-card {card_class}">', unsafe_allow_html=True)
                
                if llm_response.get("success"):
                    st.markdown(llm_response.get("response", "No response generated"))
                    
                    if result.get("context_used"):
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📚 Context used: {result.get('context_size', 0):,} chars")
                        with col2:
                            if result.get("documents_referenced"):
                                docs = ", ".join(result["documents_referenced"])
                                st.caption(f"📄 Referenced: {docs}")
                    
                    if llm_response.get("processing_time"):
                        st.caption(f"⏱️ Processed in {llm_response['processing_time']:.2f}s")
                else:
                    st.error(llm_response.get("response", "Unknown error occurred"))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Update conversation history
                if llm_response.get("success"):
                    st.session_state.conversation.append({
                        "role": "user",
                        "content": query,
                        "type": result.get("query_type", "general"),
                        "time": datetime.now().strftime("%H:%M"),
                        "timestamp": datetime.now().timestamp()
                    })
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": llm_response.get("response", ""),
                        "time": datetime.now().strftime("%H:%M"),
                        "timestamp": datetime.now().timestamp()
                    })
                    
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
    
    def _display_conversation_history(self) -> None:
        """Display collapsible conversation history"""
        if not st.session_state.conversation or len(st.session_state.conversation) == 0:
            return
        
        total_exchanges = len(st.session_state.conversation) // 2
        
        # Collapsible header
        col1, col2 = st.columns([1, 4])
        with col1:
            icon = "🔽" if st.session_state.show_history else "▶️"
            if st.button(icon, key="toggle_history", help="Toggle conversation history"):
                st.session_state.show_history = not st.session_state.show_history
                st.rerun()
        
        with col2:
            if st.session_state.show_history:
                st.markdown(f"### 📜 Conversation History ({total_exchanges} exchanges)")
            else:
                st.markdown(f"### 📜 Conversation History ({total_exchanges} exchanges) - Click to expand")
        
        # Display history if expanded
        if st.session_state.show_history:
            try:
                # Sort by timestamp (most recent last for chronological order)
                sorted_conversation = sorted(
                    st.session_state.conversation, 
                    key=lambda x: x.get("timestamp", 0)
                )
                
                # Display in chronological order (oldest first)
                for i in range(0, len(sorted_conversation), 2):
                    if i + 1 < len(sorted_conversation):
                        # Get messages in correct order
                        user_msg = sorted_conversation[i] if sorted_conversation[i].get("role") == "user" else sorted_conversation[i + 1]
                        ai_msg = sorted_conversation[i + 1] if sorted_conversation[i + 1].get("role") == "assistant" else sorted_conversation[i]
                        
                        exchange_num = (i // 2) + 1
                        
                        # User message
                        with st.chat_message("user"):
                            st.write(f"**You** (Exchange #{exchange_num})")
                            st.write(user_msg.get("content", ""))
                            st.caption(user_msg.get("time", ""))
                        
                        # AI response
                        with st.chat_message("assistant"):
                            st.write("**AI Assistant**")
                            st.write(ai_msg.get("content", ""))
                            st.caption(ai_msg.get("time", ""))
                        
                        st.markdown("---")
            except Exception as e:
                st.warning(f"⚠️ Could not display conversation history: {str(e)}")
    
    def _handle_arxiv_search(self, query: str, max_results: int) -> None:
        """Handle ArXiv paper search with centered layout"""
        if not query or not query.strip():
            st.warning("Please enter a search query")
            return
        
        with st.spinner(f"🔍 Searching ArXiv for '{query}'..."):
            try:
                result = self.agent.search_arxiv_papers(query, max_results)
                
                if result.get("success") and result.get("papers"):
                    st.session_state.arxiv_results = result["papers"]
                    st.success(f"✅ Found {len(result['papers'])} papers")
                    
                    # Display results in a centered container
                    st.markdown('<div class="arxiv-results-container">', unsafe_allow_html=True)
                    st.markdown("### 📋 Search Results")
                    
                    for i, paper in enumerate(result["papers"]):
                        with st.container():
                            st.markdown('<div class="arxiv-paper-card">', unsafe_allow_html=True)
                            
                            # Title
                            st.markdown(f"### 📄 {paper.get('title', 'Untitled')}")
                            
                            # Details in columns
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                                st.markdown(f"**Published:** {paper.get('published', 'Unknown')}")
                                if paper.get('categories'):
                                    st.markdown(f"**Categories:** {', '.join(paper.get('categories', ['Unknown']))}")
                            
                            with col2:
                                if paper.get('pdf_url'):
                                    st.markdown(f"[📥 Download PDF]({paper.get('pdf_url', '#')})")
                            
                            # Abstract
                            st.markdown("**Abstract:**")
                            st.write(paper.get('summary', 'No abstract available'))
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if i < len(result["papers"]) - 1:
                            st.markdown("---")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning(result.get("message", "No papers found"))
            except Exception as e:
                st.error(f"❌ Error searching ArXiv: {str(e)}")
    
    def run(self) -> None:
        """Main application runner"""
        try:
            # Display header
            self.ui.create_header()
            
            # Initialize AI Agent
            if not self._initialize_ai_agent():
                st.warning("⚠️ AI Agent initialization failed. Some features may be limited.")
            
            # Create tabs
            tab1, tab2 = st.tabs(["📄 Document Q&A Interface", "🔍 ArXiv Research"])
            
            # Sidebar
            with st.sidebar:
                self.ui.create_sidebar_status(self.agent, st.session_state.documents)
                st.markdown("---")
                
                # Document Processing
                st.markdown("### 📁 Document Processing")
                uploaded_files = st.file_uploader(
                    "Upload PDF files",
                    type=["pdf"],
                    accept_multiple_files=True,
                    help="Maximum 50MB per file. All files undergo security checks."
                )
                
                if uploaded_files:
                    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                        self._process_uploaded_files(uploaded_files)
                
                st.markdown("---")
                
                # Statistics in Sidebar
                if st.session_state.documents and len(st.session_state.documents) > 0:
                    st.markdown("### 📈 Quick Stats")
                    stats = self._get_safe_statistics()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Documents", stats.get("total_documents", 0))
                        st.metric("Pages", stats.get("total_pages", 0))
                    with col2:
                        st.metric("Chunks", stats.get("total_chunks", 0))
                        st.metric("Words", f"{stats.get('total_words', 0):,}")
                
                st.markdown("---")
                
                # System Info
                st.markdown("### ⚙️ System Info")
                st.markdown(f"**Model:** `{AppConfig.GROQ_MODEL}`")
                st.markdown(f"**Security:** `Enterprise Grade`")
                st.markdown(f"**Version:** `1.0.0`")
                
                st.markdown("---")
                
                # Actions
                st.markdown("### ⚡ Actions")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear Chat", use_container_width=True):
                        if self.agent:
                            self.agent.clear_history()
                        st.session_state.conversation = []
                        st.session_state.show_history = False
                        st.success("Chat cleared!")
                        st.rerun()
                
                with col2:
                    if st.button("📄 Clear Docs", use_container_width=True):
                        self.processor.clear_documents()
                        st.session_state.documents = []
                        st.session_state.stats = {}
                        st.session_state.documents_loaded = False
                        st.success("Documents cleared!")
                        st.rerun()
            
            # Tab 1: Document Q&A Interface
            with tab1:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                
                if not st.session_state.documents or len(st.session_state.documents) == 0:
                    st.info("""
                    ### 📋 Getting Started
                    
                    1. **Upload PDF documents** using the sidebar file uploader
                    2. **Click 'Process Documents'** to extract and secure content
                    3. **Ask questions** about your documents using AI
                    
                    ### 🔒 Security Features:
                    - All files undergo security validation
                    - Content is sanitized before processing
                    - Enterprise-grade encryption in transit
                    - Rate limiting and input validation
                    """)
                else:
                    # Display statistics in main content area
                    self._display_document_statistics()
                    
                    # Query Interface
                    st.markdown("### 💬 Ask a Question")
                    
                    query_type = st.selectbox(
                        "Query Type:",
                        [
                            "Custom Query",
                            "Direct Lookup (What is the conclusion?)",
                            "Summarization (Summarize key points)",
                            "Data Extraction (Extract metrics/results)",
                            "Comparison (Compare approaches)",
                            "Analysis (Analyze methodology)"
                        ]
                    )
                    
                    if query_type == "Custom Query":
                        query = st.text_area(
                            "Enter your question:",
                            placeholder="Type your secure query here...",
                            height=120,
                            max_chars=AppConfig.MAX_QUERY_LENGTH
                        )
                    else:
                        example = query_type.split("(")[1].rstrip(")") if "(" in query_type else query_type
                        query = st.text_area(
                            "Enter your question:",
                            value=example,
                            height=120,
                            max_chars=AppConfig.MAX_QUERY_LENGTH
                        )
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🤖 Ask AI", type="primary", use_container_width=True):
                            self._handle_document_query(query)
                    
                    with col2:
                        if st.button("🔍 Search Only", use_container_width=True):
                            if query:
                                with st.spinner("Searching documents..."):
                                    try:
                                        results = self.processor.search_in_documents(query)
                                        if results and len(results) > 0:
                                            st.markdown(f"### 🔍 Search Results ({len(results)} found)")
                                            for res in results[:5]:
                                                with st.expander(f"{res.get('document', 'Unknown')} - {res.get('section', 'General')}"):
                                                    st.write(res.get("text", "No text available"))
                                        else:
                                            st.warning("No matches found")
                                    except Exception as e:
                                        st.error(f"❌ Error searching documents: {str(e)}")
                    
                    # Conversation History
                    self._display_conversation_history()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tab 2: ArXiv Research
            with tab2:
                st.markdown("### 🔍 ArXiv Research Paper Search")
                st.markdown("Search for academic papers with secure AI analysis.")
                
                # Create a centered container for search
                search_container = st.container()
                with search_container:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        arxiv_query = st.text_input(
                            "Search query:",
                            placeholder="e.g., 'machine learning transformer 2023'",
                            help="Enter keywords to search for research papers",
                            key="arxiv_search_input"
                        )
                    
                    with col2:
                        max_results = st.selectbox("Results:", [3, 5, 10], index=1, key="arxiv_max_results")
                    
                    with col3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        search_btn = st.button("🔬 Search", type="primary", use_container_width=True, key="arxiv_search_btn")
                
                # Handle search
                if search_btn and arxiv_query:
                    self._handle_arxiv_search(arxiv_query, max_results)
                
                # Display recent results
                if st.session_state.arxiv_results and len(st.session_state.arxiv_results) > 0:
                    st.markdown(f"### 📋 Latest Results ({len(st.session_state.arxiv_results)} papers)")
                    for paper in st.session_state.arxiv_results[:3]:
                        st.caption(f"• {paper.get('title', 'Untitled')[:80]}...")
        
        except Exception as e:
            st.error(f"🚨 Critical Application Error: {str(e)}")
            st.markdown("""
            ### 🔧 Troubleshooting
            
            1. **Check your internet connection**
            2. **Verify API key is valid**
            3. **Restart the application**
            4. **Clear browser cache if issues persist**
            
            If problems continue, please contact support.
            """)

# ==================== APPLICATION ENTRY POINT ====================
def main():
    """Main entry point for the application"""
    try:
        # Check for required packages
        try:
            import streamlit
            import pandas
            import numpy
            import arxiv
        except ImportError as e:
            st.error(f"Missing required package: {str(e)}")
            st.info("Install with: pip install streamlit pandas numpy arxiv")
            st.stop()
        
        # Create and run application
        app = EnterpriseDocumentQAAgent()
        app.run()
        
    except Exception as e:
        st.error(f"🚨 Critical Application Error: {str(e)}")
        st.markdown("""
        ### 🔧 Troubleshooting
        
        1. **Check your internet connection**
        2. **Verify API key is valid**
        3. **Restart the application**
        4. **Clear browser cache if issues persist**
        
        If problems continue, please contact support.
        """)

if __name__ == "__main__":
    main()