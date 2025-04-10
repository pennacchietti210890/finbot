import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer, TreeSummarize
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()


def create_index(
    documents: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    embedding_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = [Document(text=doc) for doc in documents]
    index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
    return index


class RAGEngine:
    """
    Llama-index based RAG query engine.
    """

    def __init__(
        self,
        documents: List[str],
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        api_key: str = os.getenv("OPENAI_API_KEY"),
    ):
        self.index = create_index(documents)
        if llm_provider == "openai":
            self.client = OpenAI(api_key=api_key, model=model_name)
        elif llm_provider == "groq":
            self.client = Groq(api_key=api_key, model=model_name)
        elif llm_provider == "anthropic":
            self.client = Anthropic(api_key=api_key, model=model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )
        self.response_synthesizer = TreeSummarize(llm=self.client)

    def custom_query(self, query: str) -> List[str]:
        """
        Custom query method for RAG.
        """
        retrieved_nodes = self.retriever.retrieve(query)
        response = self.response_synthesizer.synthesize(
            query,
            retrieved_nodes,
        )
        return response
