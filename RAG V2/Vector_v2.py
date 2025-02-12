import re
import os
import fitz  # PyMuPDF
from tqdm.auto import tqdm
import requests

def clean_text(text: str) -> str:
    """Cleans text by removing excessive newlines and spaces."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    return text.strip()  # Trim leading/trailing spaces

def open_and_read_pdf(pdf_path: str, start_page: int = 0, stop_page: int = None):
    """
    Reads a PDF file, extracts text, cleans it, and splits it into sentences.
    
    :param pdf_path: Path to the PDF file.
    :param start_page: Number of pages to skip from the beginning.
    :param stop_page: The last page number to process (exclusive).
    :return: List of dictionaries containing page data.
    """

    # Extract just the filename from the full path
    file_name = os.path.basename(pdf_path)  # Extracts the filename from the full path

    doc = fitz.open(pdf_path)  # Open the PDF document
    pages_and_texts = []

    # Set stop_page to total pages if not specified
    if stop_page is None or stop_page > len(doc):
        stop_page = len(doc)

    for page_number in tqdm(range(start_page, stop_page), total=stop_page - start_page):  
        page = doc.load_page(page_number)  # Load page by index
        text = page.get_text("text")  # Extract text as UTF-8
        text = clean_text(text)  # Clean up extra newlines and spaces

        page_data = {
            "file_name": file_name,  # Only the filename, no path
            "page_number": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split()),
            "page_token_count": len(text) // 4,  # Approximate token count
            "text": text,
        }
        pages_and_texts.append(page_data)

    return pages_and_texts

def get_text_vectors(list_of_items: list, url_of_api: str, model_name: str):
    for item in list_of_items:
        text_for_vectoring = item["text"]

        data = {
            "model": model_name,
            "prompt": text_for_vectoring
        }

        response = requests.post(url_of_api, json=data)

        if response.status_code == 200:
            json_response = response.json()
            item["embedding"] = json_response.get("embedding", [])
        else:
            item["embedding"] = None  

    return list_of_items
