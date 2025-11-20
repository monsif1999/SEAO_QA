from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing import List, Dict
from langchain_core.embeddings.embeddings import Embeddings
from src.embedding import get_embedding_model, batch_embedding
import uuid

def get_qdrant_client() -> QdrantClient:
    try : 
        client = QdrantClient(path="./qdrant_data")
        return client
    except Exception as e : 
        print(f"A problem has occurred {e}")
        raise

def get_vector_store(embedding_model :Embeddings , client : QdrantClient, collection_name : str):
        if not client.collection_exists(collection_name):
            client.create_collection(collection_name= collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )
        try : 
            vector_store = QdrantVectorStore(client, collection_name, embedding_model)
            return vector_store
        
        except Exception as e : 
            print(f"{e}")
            raise


def index_batch(store : QdrantVectorStore, texts : List[str], ids: List[str], payloads : List[Dict]):
    try:
        uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str)) for id_str in ids]
    except Exception as e:
        print(f"Erreur lors de la conversion des IDs en UUID: {e}")
        print(f"IDs problématiques: {ids}")
        raise
    try : 
        documents_ids = store.add_texts(texts, payloads, uuid_ids)
        return documents_ids
    except Exception as e : 
        print(f"A problem occured while trying to add embedding vectors to Qdrant {e}")
        raise






