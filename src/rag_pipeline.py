from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from src.vector_db import get_qdrant_client, get_vector_store
from pydantic import BaseModel, Field

QDRANT_COLLECTION_NAME = "seao_tenders"


class Search_Input(BaseModel):
    query : str = Field(..., description="user query")


class RAGPipeline:
    def __init__(self,vector_store : QdrantVectorStore):
        self.llm_model = init_chat_model("gpt-4.1")
        self.vector_store = vector_store
        self.tools = self._get_tools()
        self.agent = create_agent(self.llm_model, self.tools, system_prompt=(
                "Tu es un expert des appels d'offres du Québec (SEAO). "
                "Ta mission est d'aider l'utilisateur en trouvant des contrats pertinents. "
                "Utilise TOUJOURS l'outil de recherche pour trouver des informations factuelles avant de répondre. "
                "Cite le nom de l'acheteur et le titre du contrat dans ta réponse."
                "Retourne toujours le lien vers l'appel ou les appels d'offres"
            ))



    def search(self, query : str, k : int = 10):
        try : 
            scored_results = self.vector_store.similarity_search_with_score(query, k)
            serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc, _ in scored_results
    )
            documents_found = []
            for doc , score in scored_results:
                doc_data = doc.metadata
                doc_data["text_content"] = doc.page_content
                doc_data["score"] = score
                documents_found.append(doc_data)
            return serialized, documents_found
        except Exception as e : 
            print(f"cant retrieve documents {e} ")
            raise
    
    def _get_tools(self):
        def _search_wrapper(query : str):
            return self.search(query)
        
        search_tool = StructuredTool.from_function(
            func=_search_wrapper,
            name = "retrive_docs",
            description="vector search and document retrieval",
            args_schema=Search_Input,
            response_format="content_and_artifact"
        )

        return [search_tool]
        
    def generate_answer(self, user_query : str):
        result = self.agent.invoke({"messages": [{"role": "user", "content": user_query}]})
        last_message = result["messages"][-1]
        response_text = last_message.content
        
        # Extraction des sources (Artifacts) depuis l'historique des messages
        sources = []
        for msg in result["messages"]:
            if hasattr(msg, "artifact") and msg.artifact:
                sources.extend(msg.artifact)
        return {
            "output": response_text,
            "sources": sources
        }
