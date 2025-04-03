"""Service classes for the FinBot application."""
from typing import Dict, Any, Optional
import logging
from app.llm.rag_query_engine import RAGEngine

logger = logging.getLogger(__name__)

# class FinBotService:
#     """
#     Singleton service for managing the LLM client and other shared resources.
    
#     This service provides a central point of access to various
#     resources used by the FinBot application, such as the LLM client.
#     """
    
#     _instance = None
#     llm_client = None
    
#     @classmethod
#     def initialize(cls, llm_client):
#         """Initialize the service with the given LLM client."""
#         if cls._instance is None:
#             cls._instance = cls()
#             cls.llm_client = llm_client
#             logger.info(f"FinBotService initialized with LLM client: {type(llm_client).__name__}")
#         else:
#             logger.info("FinBotService already initialized")
    
#     @classmethod
#     def get_instance(cls):
#         """Get the singleton instance of the service."""
#         if cls._instance is None:
#             raise RuntimeError("FinBotService has not been initialized")
#         return cls._instance
    
#     def __init__(self):
#         """Private constructor to prevent direct instantiation."""
#         if self._instance is not None:
#             raise RuntimeError("Use get_instance() to access the singleton instance")


class RAGEngineService:
    """
    Singleton service for managing RAG engines for annual reports and other documents.
    
    This service provides a central point of access to RAG engines,
    which can be initialized with company-specific documents and retrieved
    when needed by the nodes.
    """
    
    _instance = None
    _rag_engines: Dict[str, RAGEngine] = {}
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the service."""
        if cls._instance is None:
            cls._instance = cls()
            logger.info("RAGEngineService initialized")
        return cls._instance
    
    def __init__(self):
        """Private constructor to prevent direct instantiation."""
        if self._instance is not None:
            raise RuntimeError("Use get_instance() to access the singleton instance")
        self._rag_engines = {}
    
    def add_engine(self, ticker: str, documents: list, llm_provider: str = "openai", model_name: str = "gpt-4o-mini"):
        """
        Create and store a RAG engine for a specific ticker/company.
        
        Args:
            ticker: The company ticker symbol
            documents: List of documents to index
            llm_provider: The LLM provider to use
            model_name: The model name to use
        """
        logger.info(f"Creating RAG engine for {ticker}")
        self._rag_engines[ticker] = RAGEngine(
            documents=documents,
            llm_provider=llm_provider,
            model_name=model_name
        )
        logger.info(f"RAG engine for {ticker} created successfully")
        return self._rag_engines[ticker]
    
    def get_engine(self, ticker: str) -> Optional[RAGEngine]:
        """
        Get the RAG engine for a specific ticker/company.
        
        Args:
            ticker: The company ticker symbol
            
        Returns:
            The RAG engine for the specified ticker, or None if not found
        """
        engine = self._rag_engines.get(ticker)
        if engine is None:
            logger.warning(f"No RAG engine found for ticker {ticker}")
        return engine
    
    def has_engine(self, ticker: str) -> bool:
        """
        Check if a RAG engine exists for a specific ticker/company.
        
        Args:
            ticker: The company ticker symbol
            
        Returns:
            True if a RAG engine exists for the specified ticker, False otherwise
        """
        return ticker in self._rag_engines
    
    def remove_engine(self, ticker: str) -> bool:
        """
        Remove the RAG engine for a specific ticker/company.
        
        Args:
            ticker: The company ticker symbol
            
        Returns:
            True if the engine was removed, False if it didn't exist
        """
        if ticker in self._rag_engines:
            del self._rag_engines[ticker]
            logger.info(f"RAG engine for {ticker} removed")
            return True
        return False 