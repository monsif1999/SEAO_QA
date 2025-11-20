from etl.indexing import stream_table, connect_to_bigquery, process_batch
from src.vector_db import get_qdrant_client, get_vector_store, index_batch
from src.embedding import get_embedding_model


COLLECTION_NAME = "seao_infos"
BATCH_SIZE = 100


def main():
    bq_client = connect_to_bigquery()
    embedding_model = get_embedding_model("text-embedding-3-large")
    qdrant_client = get_qdrant_client()
    vs = get_vector_store(embedding_model, qdrant_client, COLLECTION_NAME)
    data_stream = stream_table(batch_size=BATCH_SIZE)
    total_docs = 0
    print(f" streaming start{BATCH_SIZE}...")
    for batch_df in data_stream : 
        try :
            documents_texte , ids_list , metadonnees_payloads =  process_batch(batch_df)
            if not documents_texte:
                continue
            index_batch(vs , documents_texte, ids_list, metadonnees_payloads)
        except Exception as e : 
            print(f"An Error has occured {e}")
            continue
        total_docs += len(documents_texte)
        print(f"Progress: {total_docs} indexed documents.")


main()