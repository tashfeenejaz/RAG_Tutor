import chromadb
from fastembed import TextEmbedding


def search(query, n_results=5, where_filter=None):
    """
    Searches the vector DB for chunks most relevant to the query.
    where_filter example: {"chapter": 4} or {"source_type": "past_paper"}
    """
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="physics_class9")

    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    query_embedding = list(embedding_model.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    return results


def print_results(results):
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        print(f"\n--- Result {i+1} (distance: {distances[i]:.4f}) ---")
        print(f"ID: {ids[i]}")
        print(f"Metadata: {metadatas[i]}")
        print(f"Text: {documents[i][:200]}...")  # first 200 chars


if __name__ == "__main__":
    # Test 1: plain semantic search, no filter
    print("=" * 50)
    print("TEST 1: 'What is Pascal's law?' (no filter)")
    print("=" * 50)
    results = search("What is Pascal's law?")
    print_results(results)

    # Test 2: filtered to only chapter 4 content
    print("\n" + "=" * 50)
    print("TEST 2: 'centre of gravity' (chapter 4 only)")
    print("=" * 50)
    results = search("centre of gravity", where_filter={"chapter": 4})
    print_results(results)

    # Test 3: filtered to only past paper MCQs
    print("\n" + "=" * 50)
    print("TEST 3: 'Newton's law' (past paper MCQs only)")
    print("=" * 50)
    results = search("Newton's law", where_filter={"question_type": "MCQ"})
    print_results(results)