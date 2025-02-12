import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_relevant_chunks(query: str, db_path: str, model_name: str = "all-mpnet-base-v2", top_k: int = 10):
    """Retrieves the most relevant text chunks from the vector database based on the query."""
    
    # Load the embedding model
    model = SentenceTransformer(model_name, device="cuda")
    query_embedding = model.encode([query])[0]
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Retrieve all stored embeddings
    cursor.execute("SELECT file_name, page_number, chunk_text, embedding FROM embeddings")
    rows = cursor.fetchall()
    
    # Compute similarity scores
    similarities = []
    for file_name, page_number, chunk_text, embedding in rows:
        stored_embedding = np.frombuffer(embedding, dtype=np.float32)
        similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        similarities.append((similarity, file_name, page_number, chunk_text))
    
    # Sort by similarity and get top_k results
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_chunks = similarities[:top_k]
    
    conn.close()
    
    return top_chunks

def generate_response(query: str, top_chunks: list, model_name: str = "google/gemma-2b-it", max_tokens: int = 8192, reserved_tokens: int = 500):
    """Generates a detailed response using google/gemma-2b-it with the retrieved chunks as context."""
    
    # Load the language model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Format the input for structured responses
    context = "\n".join([f"Section {i+1}:\n{chunk[3]}" for i, chunk in enumerate(top_chunks)])
    
    # Construct the detailed prompt
    prompt = (
    "You are an AI assistant tasked with providing detailed, structured, and informative answers. "
    "Write at least 3 paragraphs explaining this in detail."
    "Please respond in a conversational and storytelling style, as if you're explaining the topic to a colleague or friend. "
    "Your answer should be thorough and well-rounded, providing not just the facts but also background context, motivations, and implications. "
    "Break down complex concepts and characters into key ideas, offering rich detail and elaboration. Don't just state facts—provide a complete picture. "
    "Think about the nuances and dynamics that are important to understand and weave them into your answer. "
    "Take your time and ensure your response flows logically, includes plenty of examples where applicable, and makes the concepts clear and engaging. "
    "Focus on being detailed, expansive, and approachable in your explanation, just as if you were giving a deep dive on the topic.\n\n"
    
    "Use the following examples as references for the ideal answer style:\n"
    "\nExample 1:\n"
    "Query: What are the benefits of regular exercise?\n"
    "Answer: Regular exercise offers numerous benefits for both physical and mental health. It improves cardiovascular health by strengthening the heart and reducing the risk of hypertension. It also helps maintain a healthy weight by burning calories and boosting metabolism. Additionally, exercise enhances mental well-being by releasing endorphins, which are natural mood boosters. Over time, regular physical activity can improve muscle tone, increase flexibility, and promote better sleep, contributing to overall vitality. Exercise isn't just about physical health, it has positive effects on mental health too. Moreover, by developing a regular exercise routine, one can significantly reduce the risks of chronic diseases like diabetes and heart disease.\n"
    
    "\nExample 2:\n"
    "Query: What is the significance of cybersecurity in modern businesses?\n"
    "Answer: Cybersecurity is crucial for protecting a business's data, reputation, and financial assets. In today's digital age, businesses rely heavily on technology, making them vulnerable to cyberattacks. Effective cybersecurity measures safeguard sensitive customer information, such as personal data and payment details, from theft. Moreover, robust security practices ensure the integrity of business operations, preventing disruptions caused by data breaches, ransomware, or other malicious activities. A business's commitment to cybersecurity helps maintain customer trust, ensures compliance with regulations, and avoids costly disruptions that could harm its reputation. In addition, modern cybersecurity involves not just protecting data but also ensuring that businesses are ready to respond to threats in real-time, reducing downtime and preventing long-term damage.\n"
    
    "Now, please use the context below to answer the following query:\n"
    f"{context}\n\n"
    
    f"User Query: {query}\n\n"
    
    "Answer:")

    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    # Calculate the available token budget
    max_new_tokens = max_tokens - input_length - reserved_tokens
    if max_new_tokens <= 0:
        raise ValueError("The input prompt is too long. Please reduce its length.")
    
    # Generate the response
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# Example usage
db_path = "embeddings.db"
query = ""
top_chunks = get_relevant_chunks(query, db_path)
response = generate_response(query, top_chunks)

print("Generated Response:\n", response)
