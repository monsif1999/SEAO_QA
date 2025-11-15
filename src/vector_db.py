from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing import List


def get_qdrant_client() -> QdrantClient:
    try : 
        client = QdrantClient(":memory:")
        return client
    except Exception as e : 
        print(f"A problem has occurred {e}")

def get_vector_store(embeddings : List[str]  , client : QdrantClient, collection_name : str):
    vector_size = len(embeddings)

    if not client.collection_exists(collection_name):
        try : 
            client.create_collection(collection_name=collection_name,
                                     vectors_config= VectorParams(vector_size, distance = Distance.COSINE))
        except Exception as e  : 
            print(f"A problem has occurred while creating the collection {e}")

        try : 
            vector_store = QdrantVectorStore(client, collection_name, embeddings)
            return vector_store
        
        except Exception as e : 
            print(f"{e}")


