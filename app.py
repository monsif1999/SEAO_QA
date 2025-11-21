import chainlit as cl
from src.embedding import get_embedding_model
from src.vector_db import get_qdrant_client, get_vector_store
from src.rag_pipeline import RAGPipeline

# --- CONFIGURATION ---
# Doit correspondre à ta config d'indexation
COLLECTION_NAME = "seao_infos"
MODEL_NAME = "text-embedding-3-large"

@cl.on_chat_start
async def start():
    """
    Cette fonction s'exécute au démarrage de la session utilisateur.
    C'est ici qu'on initialise le 'cerveau' (RAG Pipeline).
    """
    # 1. Message d'attente
    msg = cl.Message(content="🏗️ Initialisation du système SEAO... (Connexion Qdrant & OpenAI)")
    await msg.send()

    try:
        # 2. Initialisation (Exactement comme dans ton CLI)
        embedding_model = get_embedding_model(model_name=MODEL_NAME)
        qdrant_client = get_qdrant_client()
        
        # Note : En Prod sur Hugging Face, cela se connectera au Cloud 
        # si tu as bien mis tes secrets QDRANT_URL/API_KEY.
        vector_store = get_vector_store(embedding_model, qdrant_client, COLLECTION_NAME)
        
        # 3. Création du Pipeline
        rag = RAGPipeline(vector_store=vector_store)
        
        # 4. Sauvegarde dans la session utilisateur
        # On stocke l'objet 'rag' pour pouvoir l'utiliser à chaque message
        cl.user_session.set("rag_pipeline", rag)
        
        # 5. Mise à jour du message de bienvenue
        msg.content = """👋 **Bonjour ! Je suis l'assistant expert SEAO.**
        
Posez-moi une question sur les appels d'offres du Québec.
À noter que seules les données de 2024 sont actuellement disponibles. »
*Exemple : "Quels sont les contrats de déneigement à Montréal ?"*
"""
        await msg.update()

    except Exception as e:
        msg.content = f"❌ **Erreur critique au démarrage :** \n\n{str(e)}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    """
    Cette fonction s'exécute à chaque fois que l'utilisateur envoie un message.
    """
    # 1. Récupérer le pipeline de la session
    rag = cl.user_session.get("rag_pipeline")
    
    if not rag:
        await cl.Message(content="⚠️ Le système n'est pas initialisé. Rafraîchissez la page.").send()
        return

    # 2. Envoyer un message vide pour montrer le "loader"
    msg = cl.Message(content="")
    await msg.send()

    try:
        # 3. Appeler ton Agent
        # Ton code est synchrone (def generate_answer), mais Chainlit est asynchrone.
        # On utilise cl.make_async pour ne pas bloquer l'interface pendant que GPT réfléchit.
        response_dict = await cl.make_async(rag.generate_answer)(message.content)
        
        response_text = response_dict["output"]
        sources = response_dict.get("sources", [])

        # 4. Créer les "Artifacts" (Sources affichées sur le côté)
        source_elements = []
        if sources:
            for idx, source in enumerate(sources):
                # Extraction des infos
                titre = source.get('tender_title', 'Sans titre')
                acheteur = source.get('buyer_name', 'Inconnu')
                url = source.get('source_url', '#')
                montant = source.get('total_amount')
                
                # Contenu formaté pour le panneau latéral
                content = f"**Acheteur:** {acheteur}\n\n"
                if montant:
                    content += f"**Montant:** {montant:,.2f} $\n\n"
                content += f"**Lien:** {url}\n\n"
                
                # On ajoute un extrait du texte s'il existe
                if 'text_content' in source:
                    content += f"---\n{source['text_content'][:300]}..."

                # Création de l'élément Chainlit
                element = cl.Text(
                    name=f"Source {idx + 1}", 
                    content=content, 
                    display="side" # Affiche sur le côté ("side") ou en dessous ("inline")
                )
                source_elements.append(element)
            
            # Ajout d'une petite note dans le texte principal
            response_text += f"\n\n📚 *{len(sources)} sources consultées (voir détails à côté)*"

        # 5. Envoyer la réponse finale
        msg.content = response_text
        msg.elements = source_elements
        await msg.update()

    except Exception as e:
        msg.content = f"❌ Une erreur est survenue : {str(e)}"
        await msg.update()