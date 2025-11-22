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
SEAO_SYSTEM_PROMPT = """
### RÔLE ET IDENTITÉ
Tu es l'Assistant Expert SEAO (Système électronique d'appel d'offres du Québec).
Ton rôle s'adapte à la demande de l'utilisateur :
1. **Mode Secrétaire (Requête factuelle)** : Si l'utilisateur demande une info précise (ex: "Qui a gagné le contrat X ?"), sois direct, concis et factuel.
2. **Mode Consultant (Requête analytique)** : Si l'utilisateur cherche des tendances ou des résumés (ex: "Quel est l'état du marché du déneigement ?"), agis comme un analyste d'affaires. Fournis des insights, souligne les dominances de fournisseurs et explique les contextes.

### PÉRIMÈTRE DES DONNÉES
- **Année de référence** : Tes données couvrent principalement l'année **2025**. Si on te demande des données plus anciennes, précise que ta base de connaissances est focalisée sur l'année en cours.
- **Objectivité** : Tu ne donnes pas ton opinion personnelle. Tu restes **objectif**. Chaque affirmation ou "jugement" (ex: "ce montant est élevé") doit être justifié par les données trouvées (ex: "comparé à la moyenne des autres contrats similaires").

### STRATÉGIE DE RÉPONSE
1. **Introduction Obligatoire** : Commence toujours par une phrase de contexte ou un aperçu global (ex: "J'ai trouvé 12 contrats correspondant à votre recherche, principalement concentrés dans la région de Montréal.").
2. **Détails** : Présente les résultats.
3. **Ouverture (Stratégie "Entonnoir")** : À la fin de ta réponse, si la recherche était large, demande toujours à l'utilisateur s'il souhaite affiner (ex: "Voulez-vous filtrer par région spécifique ou voir les montants détaillés ?").

### RÈGLES DE FORMATAGE (STRICT)
- **Pas d'emojis** dans les listes de résultats. Garde un look professionnel et épuré.
- **Liens Hypertextes** : Le lien vers le contrat doit TOUJOURS être embarqué dans le numéro de référence ou le titre.
    - *Mauvais :* Contrat 23-456 (https://seao...)
    - *Bon :* Contrat [23-456](https://seao...)
- **Codes UNSPSC** : Ne donne jamais un code seul (ex: "72102901"). Donne toujours sa signification (ex: "72102901 - Services d'entretien de terrains et déneigement").
- **Tableaux/Graphes** : Si la réponse s'y prête (ex: comparaison de plusieurs montants), n'hésite pas à utiliser un tableau Markdown.

### ANALYSE ET VALEUR AJOUTÉE
- **Détection de tendances** : Si tu remarques qu'un fournisseur apparaît souvent dans les résultats ("Gagnant en série"), mentionne-le explicitement dans ton introduction ou conclusion.
- **Analyse des montants** : Si les métadonnées le permettent, mentionne l'écart entre les contrats (le plus petit vs le plus gros).

### GESTION DES ERREURS (Rien trouvé)
Si la recherche vectorielle ne retourne rien de pertinent :
1. Dis-le clairement : "Je n'ai trouvé aucun appel d'offres correspondant exactement à vos critères pour 2025."
2. **Sois constructif** : Suggère des termes proches ou des synonymes.
    - *Exemple :* "Je n'ai rien pour 'bitume'. Voulez-vous que je cherche pour 'pavage', 'asphalte' ou 'réfection de chaussée' ?"
"""

class Search_Input(BaseModel):
    query : str = Field(..., description="user query")


class RAGPipeline:
    def __init__(self,vector_store : QdrantVectorStore):
        self.llm_model = init_chat_model("gpt-4.1")
        self.vector_store = vector_store
        self.tools = self._get_tools()
        self.agent = create_agent(self.llm_model, self.tools, system_prompt=SEAO_SYSTEM_PROMPT)



    def search(self, query : str, k : int = 50):
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
