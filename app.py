import os
import sys

sys.path.append(os.getcwd())

from src.embedding import get_embedding_model
from src.vector_db import get_qdrant_client, get_vector_store
from src.rag_pipeline import RAGPipeline

COLLECTION_NAME = "seao_infos"
MODEL_NAME = "text-embedding-3-large" 

def main():
    print("Initialisation du système RAG SEAO...")

    try:
        embedding_model = get_embedding_model(model_name=MODEL_NAME)
        
        try : 
            print("connexion au database")
            qdrant_client = get_qdrant_client()
            print("fini")
        except Exception as e : 
            print(e)
            raise
        vector_store = get_vector_store(embedding_model, qdrant_client, COLLECTION_NAME)

        rag = RAGPipeline(vector_store=vector_store)
        
        print(f"✅ Système prêt ! Connecté à la collection '{COLLECTION_NAME}'.")
        print("----------------------------------------------------------------")
        print("Tapez 'exit', 'quit' ou 'q' pour quitter.")

    except Exception as e:
        print(f" Erreur fatale au démarrage : {e}")
        return

    while True:
        try:
            user_input = input("\n Votre question : ")

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Au revoir !")
                break
            
            if not user_input.strip():
                continue

            print("L'agent réfléchit...")

            result = rag.generate_answer(user_input)

            print(f"\n💡 RÉPONSE :\n{result['output']}\n")

            if result.get('sources'):
                print("SOURCES CONSULTÉES :")
                for source in result['sources']:
                    titre = source.get('tender_title', 'Sans titre')
                    acheteur = source.get('buyer_name', 'Acheteur inconnu')
                    url = source.get('source_url', '#')
                    
                    print(f"   {titre}")
                    print(f"    {acheteur} | 🔗 {url}")
            else:
                print(" (Aucune source spécifique renvoyée par l'outil)")

            print("-" * 60)

        except Exception as e:
            print(f"Erreur pendant la requête : {e}")
            continue

main()