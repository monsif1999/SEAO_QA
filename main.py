import os, json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain.tools import tool


# ========== CONFIG ==========
JSON_PATH = "./data/hebdo_20241230_20250105.json"
COLLECTION = "example_collection"
PERSIST_DIR = "./chroma_langchain_db"
EMBED_MODEL = "text-embedding-3-large"   # ou "text-embedding-3-small" pour réduire le coût
MAX_TOKENS_PER_REQ = 250_000             # marge de sécurité < 300k
CHUNK_TOKENS = 1200
CHUNK_OVERLAP = 150

# ========== INIT ==========
load_dotenv()
model = init_chat_model("gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

enc = tiktoken.get_encoding("cl100k_base")
tok_len = lambda s: len(enc.encode(s))

# ========== SECTION A: texte compact par release ==========
def record_to_text(r: dict) -> str:
    t = r.get("tender") or {}
    items = t.get("items") or []
    first_item = items[0] if items else {}
    cls = first_item.get("classification") or {}
    parties = r.get("parties") or []
    addr = (parties[0].get("address") if parties else {}) or {}
    url = (t.get("documents") or [{}])[0].get("url")

    lines = [
        f"Titre: {t.get('title','')}",
        f"UNSPSC: {cls.get('id','')} ({cls.get('description','')})",
        f"Acheteur: {(r.get('buyer') or {}).get('name','')}",
        f"Localisation: {addr.get('locality','')}, {addr.get('region','')} {addr.get('postalCode','')}",
        f"Description: {first_item.get('description','')}",
        f"OCID: {r.get('ocid','')}",
        f"Date: {r.get('date','')}",
        f"URL: {url or ''}",
    ]
    # Texte court et informatif: évite d'inclure l'objet JSON entier
    return "\n".join(lines)

# ========== SECTION B: chargement + split token-aware ==========
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
releases = data.get("releases") or []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TOKENS,      # on contrôle par tokens via length_function
    chunk_overlap=CHUNK_OVERLAP,
    length_function=tok_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

docs = []
for r in releases:
    txt = record_to_text(r)
    # si texte ultra court, inutile de splitter
    if tok_len(txt) <= CHUNK_TOKENS:
        docs.append(
            Document(
                page_content=txt,
                metadata={
                    "ocid": r.get("ocid"),
                    "id": r.get("id"),
                    "date": r.get("date"),
                    "unspsc": ((r.get("tender") or {}).get("items") or [{}])[0].get("classification",{}).get("id"),
                    "buyer": (r.get("buyer") or {}).get("name"),
                    "language": r.get("language"),
                },
            )
        )
    else:
        for chunk in splitter.split_text(txt):
            # petit garde-fou si jamais un chunk dépasse 8k tokens
            if tok_len(chunk) > 8192:
                chunk = enc.decode(enc.encode(chunk)[:8192])
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "ocid": r.get("ocid"),
                        "id": r.get("id"),
                        "date": r.get("date"),
                        "unspsc": ((r.get("tender") or {}).get("items") or [{}])[0].get("classification",{}).get("id"),
                        "buyer": (r.get("buyer") or {}).get("name"),
                        "language": r.get("language"),
                    },
                )
            )

# ========== SECTION C: batching embeddings ≤ 250k tokens ==========
def batch_by_token_budget(documents, max_tokens=MAX_TOKENS_PER_REQ):
    batch, cur_tokens = [], 0
    for d in documents:
        t = tok_len(d.page_content)
        # si un seul doc dépasse la limite (rare), on le tronque à ~8k tokens
        if t > 8192:
            d = Document(page_content=enc.decode(enc.encode(d.page_content)[:8192]), metadata=d.metadata)
            t = tok_len(d.page_content)
        # si ajouter ce doc dépasse le budget → on yield le batch courant
        if batch and cur_tokens + t > max_tokens:
            yield batch
            batch, cur_tokens = [], 0
        batch.append(d)
        cur_tokens += t
    if batch:
        yield batch

inserted_ids = []
for bdocs in batch_by_token_budget(docs):
    # Chroma.add_documents fera 1 appel d'embedding pour CE lot seulement
    ids = vector_store.add_documents(bdocs)
    inserted_ids.extend(ids)

print(f"Inserted {len(inserted_ids)} chunks. First 3 ids: {inserted_ids[:3]}")

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
     """
Tu es un agent intelligent spécialisé dans l’analyse et la recherche d’informations
dans les appels d’offres publics du Québec (SEAO).

Tu disposes d’un outil appelé `retrieve_context` qui te permet de récupérer
des passages de texte pertinents à partir d’une base vectorielle (embeddings).

Ton objectif est d’utiliser cet outil pour :
- Trouver et résumer les informations pertinentes à la requête de l’utilisateur.
- Citer les acheteurs, les dates, les montants et les codes UNSPSC lorsqu’ils sont disponibles.
- Expliquer ta réponse clairement en français, de façon concise et structurée.
- Si l’information n’est pas trouvée dans le contexte, indique-le explicitement
  (ne devine pas ni n’invente de données).

Règles importantes :
- Utilise `retrieve_context` avant de répondre.
- Ne retourne jamais de JSON brut, mais un texte lisible et bien formaté.
- Si plusieurs appels d’offres sont pertinents, donne les plus récents ou les plus proches
  de la requête, avec une brève synthèse.

Exemples de requêtes utilisateur :
- “Quels appels d’offres concernent le déneigement à Laval ?”
- “Donne-moi le contrat du CIUSSS du Saguenay pour pompes à perfusion.”
- “Quels organismes ont attribué un contrat de gré à gré en avril 2025 ?”
"""
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "Je cherche les appels d offres qui ont été publié sur la ville de Montréal et retourne moi le titre et l'organisme qui a publié l'offre ainsi que la date \n\n"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()