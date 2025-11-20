import os 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.embeddings.embeddings import Embeddings
from typing import List

load_dotenv()



def get_embedding_model(model_name : str) -> Embeddings :
    open_ai_key = os.getenv("OPENAI_API_KEY")
    if not open_ai_key:
        raise ValueError("OPENAI API KEY is not found in .env")
    
    try : 
        embedding = OpenAIEmbeddings(model = model_name)
        return embedding  

    except Exception as e:
        print(f"An unexpected error occurred : {e}")



def batch_embedding(model : Embeddings, batch_texts : List[str]):
    try : 
        vectors = model.embed_documents(batch_texts)
        return vectors
    except Exception as e : 
        print(f"A problem has occurred while trying to embedd documents : {e}")

