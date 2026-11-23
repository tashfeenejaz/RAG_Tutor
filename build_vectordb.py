"""Embed all chunks with BGE-small and store them in ChromaDB (safe to re-run)."""
import json

import chromadb
from fastembed import TextEmbedding

import config


def load_jsonl(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def sanitize_metadata(metadata):
    """ChromaDB rejects None values (e.g. chapter=None on cover pages)."""
    return {k: ("unknown" if v is None else v) for k, v in metadata.items()}


def build_vector_db(batch_size=100):
    all_chunks = load_jsonl(config.TEXTBOOK_CHUNKS) + load_jsonl(config.PAST_PAPER_CHUNKS)
    ids = [c["id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk ids - re-run the chunking scripts first."
    print(f"Total chunks to embed: {len(all_chunks)}")

    model = TextEmbedding(model_name=config.EMBED_MODEL)
    documents = [c["text"] for c in all_chunks]
    metadatas = [sanitize_metadata(c["metadata"]) for c in all_chunks]

    print("Generating embeddings...")
    embeddings = [e.tolist() for e in model.embed(documents)]

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    # Rebuild from scratch so re-running never leaves stale or duplicate rows.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=config.COLLECTION_NAME)

    print("Inserting into ChromaDB...")
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  Inserted {min(i + batch_size, len(ids))}/{len(ids)}")

    print(f"\nDone. Collection '{config.COLLECTION_NAME}' has {collection.count()} items.")


if __name__ == "__main__":
    build_vector_db()
