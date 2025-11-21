from etl.indexing import stream_table, connect_to_bigquery, process_batch
from src.vector_db import get_qdrant_client, get_vector_store, index_batch
from src.embedding import get_embedding_model
import os
import logging
from datetime import datetime

COLLECTION_NAME = "seao_infos"
BATCH_SIZE = 100
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/indexing_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # Écrit dans le fichier
        logging.StreamHandler()            # Affiche dans le terminal
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info(f"🚀 Démarrage du pipeline d'indexation. Batch size: {BATCH_SIZE}")
    logger.info(f"📁 Les logs seront enregistrés dans : {log_filename}")
    bq_client = connect_to_bigquery()
    logger.info("🧠 Chargement du modèle d'embedding (text-embedding-3-large)...")
    embedding_model = get_embedding_model("text-embedding-3-large")
    logger.info("🗄️ Connexion à Qdrant...")
    qdrant_client = get_qdrant_client()
    logger.info(f"📦 Initialisation du Vector Store (Collection: {COLLECTION_NAME})...")
    vs = get_vector_store(embedding_model, qdrant_client, COLLECTION_NAME)
    logger.info("🌊 Démarrage du stream de données...")
    data_stream = stream_table(batch_size=BATCH_SIZE)
    total_docs = 0
    batch_count = 0
    print(f" streaming start{BATCH_SIZE}...")
    for batch_df in data_stream : 
        batch_count += 1
        try :
            documents_texte , ids_list , metadonnees_payloads =  process_batch(batch_df)
            if not documents_texte:
                continue
            index_batch(vs , documents_texte, ids_list, metadonnees_payloads)
        except Exception as e : 
            print(f"An Error has occured {e}")
            continue
        current_batch_len = len(documents_texte)
        total_docs += len(documents_texte)
        logger.info(f"✅ Batch {batch_count} indexé ({current_batch_len} docs). Total: {total_docs}")
    logger.info(f"🎉 Indexation terminée avec succès ! Total documents: {total_docs}")

main()