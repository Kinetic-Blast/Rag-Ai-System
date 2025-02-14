import Vector_v2
import SearchDataEmbed
import database_commands
import json
import tiktoken


def add_complete_book(db_name: str, pdf_path: str, url_of_api: str, model_name: str, start_page: int = 0, stop_page: int = None):
    """Extracts, processes, and adds a complete book to the database."""
    
    # Step 1: Extract text from the PDF
    pages = Vector_v2.open_and_read_pdf(pdf_path,start_page,stop_page)

    if not pages:
        return "No pages extracted from the PDF."

    # Step 2: Generate embeddings
    pages = Vector_v2.get_text_vectors(pages, url_of_api, model_name)

    # Step 3: Add the book to the database
    book_name = SearchDataEmbed.os.path.basename(pdf_path)  # Use filename as book name
    database_commands.add_book(db_name, book_name, book_name)

    # Step 4: Store pages in the database
    for page in pages:
        database_commands.add_page(db_name, page)

    return f"Successfully added {book_name} with {len(pages)} pages."


def list_models(url_of_api:str):
    response = Vector_v2.requests.get(url_of_api)
    if response.status_code == 200:
        # Parse the response JSON
        models_data = response.json()
    
        # Extract the model IDs into an array
        model_ids = [model['id'] for model in models_data['data'] if model['id'] != "nomic-embed-text:latest"] #hide ebedding model
    
        return(model_ids)
    else:
        return(f"Failed to retrieve models. Status code: {response.status_code}")
    
def count_tokens(data_list, model="gpt-3.5-turbo"):
    """Counts the total number of tokens in an array of dictionaries containing 'query' and 'response' keys."""
    enc = tiktoken.encoding_for_model(model)
    
    total_tokens = 0
    for data in data_list:
        if isinstance(data, dict):  # Ensure it's a dictionary
            for key in ['query', 'response']:  # Process only 'query' and 'response' keys
                if key in data and isinstance(data[key], str):  
                    total_tokens += len(enc.encode(data[key]))
    
    return total_tokens


def query_ai_system(url_of_api:str, query:str, model:str,rag_items = [], memory=[]):
    # Format the retrieved context


    formatted_context = "\n\n".join(
        [f"Source {i+1}:\n{item}" for i, item in enumerate(rag_items)]
    ) if rag_items else "No Context Provided."



    formatted_memory = "\n\n".join(
        [f"Query: {mem['query']}\nResponse: {mem['response']}" for mem in memory]
    ) if memory else "No prior memory."

    # Define the structured prompt
    prompt = f"""
You are a helpful assistant that retrieves relevant information but explains it in a natural, engaging way.


Here’s what you need to do:  
1. Summarize the most important points from the retrieved info.  
2. Explain it casually, like you’re talking to a friend.  
3. Keep it structured but easy to follow.  
4. Avoid Direct References to Sources: Don’t refer to specific sources or data from which the information is retrieved. Frame your response to stand on its own, as if it’s based on the analysis or synthesis of the information without pointing out its origins.
Now, craft a response that is helpful and natural.
5.Maintain Relevance and Accuracy: Make sure that every part of your response is relevant to the user’s query. Stay focused on what is most helpful to the user without going off-topic.


### Memory:
{formatted_memory}

### Retrieved Context:
{formatted_context}

### User Query:
{query}

### Response:
"""

    # API Request to Ollama
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Change to True if you want a streaming response
    }

    headers = {"Content-Type": "application/json"}
    response = Vector_v2.requests.post(url_of_api, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        response_text = response.json().get("response", "No response received.")
        memory.append({"query": query, "response": response_text})  # Store query-response pair in memory
        return response_text, memory
    else:
        return f"Error: {response.status_code} - {response.text}", memory

def list_dbs(directory="."):
    """Lists all .db files in the specified directory."""
    return [f for f in SearchDataEmbed.os.listdir(directory) if f.endswith(".db")]





