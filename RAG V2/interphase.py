import os
import Vector_v2
import SearchDataEmbed




def add_complete_book(db_name: str, pdf_path: str, url_of_api: str, model_name: str, start_page: int = 0, stop_page: int = None):
    """Extracts, processes, and adds a complete book to the database."""
    
    # Step 1: Extract text from the PDF
    pages = Vector_v2.open_and_read_pdf(pdf_path,start_page,stop_page)

    if not pages:
        return "No pages extracted from the PDF."

    # Step 2: Generate embeddings
    pages = Vector_v2.get_text_vectors(pages, url_of_api, model_name)

    # Step 3: Add the book to the database
    book_name = os.path.basename(pdf_path)  # Use filename as book name
    SearchDataEmbed.database_commands.add_book(db_name, book_name, book_name)

    # Step 4: Store pages in the database
    for page in pages:
        SearchDataEmbed.database_commands.add_page(db_name, page)

    return f"Successfully added {book_name} with {len(pages)} pages."


def list_models(url_of_api:str):
    response = Vector_v2.requests.get(url_of_api)
    if response.status_code == 200:
        # Parse the response JSON
        models_data = response.json()
    
        # Extract the model IDs into an array
        model_ids = [model['id'] for model in models_data['data'] if model['id'] != "nomic-embed-text:latest"] #hide ebedding model
    
        print(model_ids)
    else:
        print(f"Failed to retrieve models. Status code: {response.status_code}")


list_models("http://192.168.50.61:11434/v1/models")



'''
query ai
clean memory
update memory
maintain memory (128K tokens)


'''

