import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import tempfile
import shutil

from app.finbot.services import RAGEngineService
from app.llm.rag_query_engine import RAGEngine


@pytest.fixture
def reset_singleton():
    """Reset the RAGEngineService singleton before each test."""
    RAGEngineService._instance = None
    yield
    RAGEngineService._instance = None


@pytest.fixture
def mock_query_engine():
    """Fixture providing a mock RAGEngine."""
    mock_engine = MagicMock(spec=RAGEngine)
    mock_engine.custom_query.return_value = "This is a test response from the RAG engine"
    return mock_engine


def test_singleton_pattern(reset_singleton):
    """Test that RAGEngineService follows the singleton pattern."""
    service1 = RAGEngineService.get_instance()
    service2 = RAGEngineService.get_instance()
    
    assert service1 is service2
    assert isinstance(service1, RAGEngineService)


def test_get_engine_nonexistent(reset_singleton):
    """Test getting a non-existent engine returns None."""
    service = RAGEngineService.get_instance()
    engine = service.get_engine("NONEXISTENT")
    
    assert engine is None


def test_set_and_get_engine(reset_singleton, mock_query_engine):
    """Test setting and getting an engine."""
    service = RAGEngineService.get_instance()
    
    # Set the engine directly
    service._rag_engines["TEST"] = mock_query_engine
    
    # Get the engine
    engine = service.get_engine("TEST")
    
    # Verify it's the same engine
    assert engine is mock_query_engine


def test_has_engine(reset_singleton, mock_query_engine):
    """Test checking if an engine exists."""
    service = RAGEngineService.get_instance()
    
    assert not service.has_engine("TEST")
    
    # Set the engine directly
    service._rag_engines["TEST"] = mock_query_engine
    
    assert service.has_engine("TEST")


def test_query_engine(reset_singleton, mock_query_engine):
    """Test querying an engine."""
    service = RAGEngineService.get_instance()
    # Set the engine directly
    service._rag_engines["TEST"] = mock_query_engine
    
    # We need to create a mock object to simulate the return of custom_query
    mock_response = MagicMock()
    mock_response.response = "This is a test response from the RAG engine"
    mock_query_engine.custom_query.return_value = mock_response
    
    # Get the engine and query it directly as there's no query_engine method
    engine = service.get_engine("TEST")
    result = engine.custom_query("What is the revenue?")
    
    assert result.response == "This is a test response from the RAG engine"
    mock_query_engine.custom_query.assert_called_once_with("What is the revenue?")


def test_query_nonexistent_engine(reset_singleton):
    """Test querying a non-existent engine returns None."""
    service = RAGEngineService.get_instance()
    
    result = service.get_engine("NONEXISTENT")
    
    assert result is None


@patch('app.finbot.services.RAGEngine')
def test_add_engine(mock_rag_engine_class, reset_singleton):
    """Test adding an engine."""
    # Mock the RAGEngine class
    mock_engine = MagicMock()
    mock_rag_engine_class.return_value = mock_engine
    
    # Create service and add engine
    service = RAGEngineService.get_instance()
    result = service.add_engine("TEST", ["This is test content"])
    
    # Verify results
    assert result is mock_engine
    mock_rag_engine_class.assert_called_once()
    assert service.get_engine("TEST") is mock_engine


@patch('app.finbot.services.RAGEngine')
def test_add_engine_with_custom_params(mock_rag_engine_class, reset_singleton):
    """Test adding an engine with custom parameters."""
    # Mock the RAGEngine class
    mock_engine = MagicMock()
    mock_rag_engine_class.return_value = mock_engine
    
    # Create service and add engine with custom params
    service = RAGEngineService.get_instance()
    result = service.add_engine(
        "TEST", 
        ["This is test content"], 
        llm_provider="groq", 
        model_name="llama3-70b-8192"
    )
    
    # Verify results
    assert result is mock_engine
    mock_rag_engine_class.assert_called_once_with(
        documents=["This is test content"],
        llm_provider="groq",
        model_name="llama3-70b-8192"
    )
    assert service.get_engine("TEST") is mock_engine


def test_remove_engine(reset_singleton, mock_query_engine):
    """Test removing an engine."""
    service = RAGEngineService.get_instance()
    # Set the engine directly
    service._rag_engines["TEST"] = mock_query_engine
    
    assert service.has_engine("TEST")
    
    service.remove_engine("TEST")
    
    assert not service.has_engine("TEST") 