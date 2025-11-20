import os
import sys

# Ajoute le dossier courant au chemin python pour trouver les modules 'src'
sys.path.append(os.getcwd())

from src.embedding import get_embedding_model
from src.vector_db import get_qdrant_client, get_vector_store
from src.rag_pipeline import RAGPipeline

# --- CONFIGURATION ---
# Doit correspondre exactement à ce que tu as utilisé pour l'indexation
COLLECTION_NAME = "seao_infos"
MODEL_NAME = "text-embedding-3-large" 

def main():
    print("🏗️ Initialisation du système RAG SEAO...")

    try:
        # 1. Initialisation des composants
        # On charge le modèle d'embedding
        embedding_model = get_embedding_model(model_name=MODEL_NAME)
        
        # On se connecte à Qdrant (sur le disque)
        try : 
            print("connexion au database")
            qdrant_client = get_qdrant_client()
            print("fini")
        except Exception as e : 
            print(e)
            raise
        # On charge le Vector Store (la collection doit exister)
        vector_store = get_vector_store(embedding_model, qdrant_client, COLLECTION_NAME)

        # On initialise le cerveau (Pipeline)
        rag = RAGPipeline(vector_store=vector_store)
        
        print(f"✅ Système prêt ! Connecté à la collection '{COLLECTION_NAME}'.")
        print("----------------------------------------------------------------")
        print("Tapez 'exit', 'quit' ou 'q' pour quitter.")

    except Exception as e:
        print(f"❌ Erreur fatale au démarrage : {e}")
        return

    # 2. Boucle de discussion
    while True:
        try:
            # Input utilisateur
            user_input = input("\n❓ Votre question : ")

            # Gestion de la sortie
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Au revoir !")
                break
            
            # Ignorer les entrées vides
            if not user_input.strip():
                continue

            print("🤖 L'agent réfléchit...")

            # Appel au Pipeline RAG
            # result est un dictionnaire : {'output': str, 'sources': list}
            result = rag.generate_answer(user_input)

            # Affichage de la réponse
            print(f"\n💡 RÉPONSE :\n{result['output']}\n")

            # Affichage des sources
            if result.get('sources'):
                print("📚 SOURCES CONSULTÉES :")
                for source in result['sources']:
                    # On utilise .get() pour éviter les erreurs si une clé manque
                    titre = source.get('tender_title', 'Sans titre')
                    acheteur = source.get('buyer_name', 'Acheteur inconnu')
                    url = source.get('source_url', '#')
                    
                    print(f"   - {titre}")
                    print(f"     🏢 {acheteur} | 🔗 {url}")
            else:
                print("ℹ️ (Aucune source spécifique renvoyée par l'outil)")

            print("-" * 60)

        except Exception as e:
            print(f"❌ Erreur pendant la requête : {e}")
            # On continue la boucle pour ne pas crasher l'app
            continue

main()