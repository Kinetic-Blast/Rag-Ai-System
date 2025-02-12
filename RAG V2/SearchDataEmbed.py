import sqlite3
import numpy as np
import requests
import database_commands  # Import the database functions

def get_query_embedding(query: str, url_of_api: str, model_name: str):
    """Get the embedding vector for a given query string."""
    data = {"model": model_name, "prompt": query}
    response = requests.post(url_of_api, json=data)
    
    if response.status_code == 200:
        return np.array(response.json().get("embedding", []), dtype=np.float32)
    else:
        return None

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return -1  # Return low score for invalid vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_pages(db_name: str, query: str, url_of_api: str, model_name: str, top_n: int = 10, focus_only: bool = False):
    """Find the top N most similar pages to the query based on cosine similarity, with optional focus filtering."""
    
    query_embedding = get_query_embedding(query, url_of_api, model_name)
    if query_embedding is None:
        return []

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Retrieve pages but exclude pages from excluded books
    if focus_only:
        cursor.execute("""
        SELECT p.id, p.book_id, p.page_number, p.text, p.embedding 
        FROM pages p 
        JOIN books b ON p.book_id = b.id 
        WHERE b.excluded = 0 AND b.focused = 1
        """)
    else:
        cursor.execute("""
        SELECT p.id, p.book_id, p.page_number, p.text, p.embedding 
        FROM pages p 
        JOIN books b ON p.book_id = b.id 
        WHERE b.excluded = 0
        """)

    pages = cursor.fetchall()
    conn.close()

    results = []
    
    for page in pages:
        page_id, book_id, page_number, text, embedding_blob = page
        if embedding_blob is None:
            continue  # Skip pages without embeddings

        # Convert BLOB back to numpy array
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        # Compute similarity
        similarity = cosine_similarity(query_embedding, embedding)

        results.append((page_id, book_id, page_number, similarity, text))

    # Sort by similarity (highest first) and return top N
    results.sort(key=lambda x: x[3], reverse=True)
    return results[:top_n]

def search_and_display_results(db_name: str, query: str, url_of_api: str, model_name: str, top_n: int = 10, focus_only: bool = False):
    """Search for the most relevant pages and display them in a readable format."""
    top_matches = find_similar_pages(db_name, query, url_of_api, model_name, top_n, focus_only)

    if not top_matches:
        return None

    return top_matches



