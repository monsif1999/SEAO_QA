import os
from google.cloud import bigquery
from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError
import pandas as pd
from typing import Generator, Tuple, List, Dict, Any

load_dotenv()
project_id = os.getenv("PROJECT_ID")
dataset_id = os.getenv("DATASET_ID")
table_id = os.getenv("TABLE_ID")


def connect_to_bigquery() -> bigquery.Client:
    if not project_id:
        raise ValueError("PROJECT_ID not found in .env")
    try : 
        client = bigquery.Client(project = project_id)
        print("Login Successful..")
        return client
    except GoogleAPIError as e :
        print(f"Connection Error {e}")
        raise


def stream_table(batch_size : int = 100):
    client = connect_to_bigquery()
    
    table_path = f"{project_id}.{dataset_id}.{table_id}"
    query = f"""
        SELECT * FROM `{table_path}` 
WHERE start_date >= '2024-01-01' AND start_date <= '2024-12-31'
        ORDER BY start_date DESC
    """
    if not table_id:
        raise ValueError("table_path not found in .env")
    try : 
        query_job = client.query(query)
        rows_iter = query_job.result(page_size=batch_size)
        batch_rows = []
        for row in rows_iter:
            batch_rows.append(dict(row))
            if len(batch_rows) >= batch_size:
                yield pd.DataFrame(batch_rows)
                batch_rows = []
        #load the the remaining batch 
        if batch_rows:
            yield  pd.DataFrame(batch_rows)
    except GoogleAPIError as e :
        print(f"Cant get the data from table {e}")
        raise


    
def process_batch(batch_df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Traite un DataFrame (lot) et retourne les documents texte et les métadonnées.
    """
    
    documents_texte = []
    metadonnees_payloads = []
    ids_list = []
    
    for index, row in batch_df.iterrows():
        
        texte_parts = []
        
        if pd.notna(row['tender_title']):
            texte_parts.append(f"Appel d'offres : {row['tender_title']}")
        if pd.notna(row['tender_id']):
            texte_parts.append(f"Numéro d'appel d'offres : {row['tender_id']}")
            
        if pd.notna(row['buyer_name']):
            texte_parts.append(f"Publié par : {row['buyer_name']} ({row.get('buyer_locality', 'N/A')}).")
        
        if pd.notna(row['status']):
            texte_parts.append(f"Statut : {row['status']}.")
        
        if pd.notna(row['start_date']):
            texte_parts.append(f"Date de début : {row['start_date']}")
        if pd.notna(row['end_date']):
            texte_parts.append(f"Date de fin : {row['end_date']}")

        if pd.notna(row['procurement_method_details']):
            texte_parts.append(f"Méthode d'approvisionnement : {row['procurement_method_details']}.")
            
        if pd.notna(row['procurement_method_rationale']):
            texte_parts.append(f"Justification : {row['procurement_method_rationale']}.")
            
        if pd.notna(row['unspsc_description']):
            texte_parts.append(f"Description : {row['unspsc_description']}.")

        if pd.notna(row['total_award_amount']) and row['total_award_amount'] > 0:
            texte_parts.append(f"Montant total adjugé : {row['total_award_amount']:.2f} CAD.")
            if pd.notna(row['main_supplier']):
                texte_parts.append(f"Fournisseur principal : {row['main_supplier']}.")
        
        if pd.notna(row['last_contract_status']):
            texte_parts.append(f"Statut du contrat : {row['last_contract_status']}.")
            
        if pd.notna(row['seao_url']):
            texte_parts.append(f"Lien SEAO : {row['seao_url']}")

        texte = "\n".join(texte_parts)
        
        
        metadata = {
           "ocid": row['ocid'],
           "tender_id": row['tender_id'],
           "source_url": row.get('seao_url'),
           "buyer_name": row.get('buyer_name'),
           "buyer_locality": row.get('buyer_locality'),
           "is_municipal": row.get('is_municipal'),
           "status": row.get('status'),
           "procurement_method": row.get('procurement_method'),
           "unspsc_code": row.get('unspsc_code'),
           "total_amount": float(row['total_award_amount']) if pd.notna(row['total_award_amount']) else None,
           "supplier": row.get('main_supplier'),
           "start_date": str(row['start_date']) if pd.notna(row['start_date']) else None,
           "end_date": str(row['end_date']) if pd.notna(row['end_date']) else None,
           "contract_signed_date": str(row['last_contract_signed_date']) if pd.notna(row['last_contract_signed_date']) else None
        }
        
        documents_texte.append(texte)
        metadonnees_payloads.append(metadata)
        ids_list.append(str(row['ocid']))

    return documents_texte, ids_list , metadonnees_payloads
    
