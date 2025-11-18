from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing import List, Dict
from langchain_core.embeddings.embeddings import Embeddings
from embedding import get_embedding_model, batch_embedding
import uuid

def get_qdrant_client() -> QdrantClient:
    try : 
        client = QdrantClient(":memory:")
        return client
    except Exception as e : 
        print(f"A problem has occurred {e}")
        raise

def get_vector_store(embedding_model :Embeddings , client : QdrantClient, collection_name : str):
        if not client.collection_exists(collection_name):
            client.create_collection(collection_name= collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
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






embedding_model = get_embedding_model(model_name="text-embedding-3-small")
qdrant_client = get_qdrant_client()
vs = get_vector_store(embedding_model, qdrant_client, collection_name= 'TEST')
test_texts = [
            "Contrat de déneigement pour la ville de Montréal",
            "Achat d'équipement informatique pour le Collège Ahuntsic"
        ]
test_payloads = [
            {"ocid": "ocds-123", "buyer_name": "Test Buyer 1"},
            {"ocid": "ocds-456", "buyer_name": "Test Buyer 2"}
        ]
test_ids = [p['ocid'] for p in test_payloads]
        

document_ids = index_batch(vs, test_texts, test_ids,test_payloads)
print(document_ids)
