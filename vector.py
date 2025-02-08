import os
import sqlite3
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import re
from tqdm.auto import tqdm

MAX_CHUNK_SIZE = 384
BATCH_SIZE = 32  # Adjust based on your GPU memory

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def split_into_sentences(text: str):
    """Splits text into sentences using regex."""
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

def chunk_sentences(sentences: list, max_chunk_size: int) -> list[list[str]]:
    """Splits sentences into chunks, ensuring no sentence is fragmented."""
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_chunk_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_chunk_size += sentence_size
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [sentence]
            current_chunk_size = sentence_size
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Opens a PDF file, reads its text content page by page, and collects statistics."""
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), desc="Reading PDF pages"):
        text = page.get_text()
        text = text_formatter(text)
        sentences = split_into_sentences(text)
        sentence_chunks = chunk_sentences(sentences, MAX_CHUNK_SIZE)
        pages_and_texts.append({
            "page_number": page_number,
            "sentence_chunks": sentence_chunks
        })
    return pages_and_texts

def process_pdf_files(directory: str, db_path: str):
    """Processes all PDF files in a directory and stores embeddings in an SQLite database."""
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            page_number INTEGER,
            chunk_text TEXT,
            embedding BLOB,
            UNIQUE(file_name, page_number, chunk_text)
        )
    ''')

    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    for file_name in tqdm(pdf_files, desc="Processing PDF files"):
        pdf_path = os.path.join(directory, file_name)
        pages_and_texts = open_and_read_pdf(pdf_path)

        all_chunks = []
        for item in pages_and_texts:
            for sentence_chunk in item["sentence_chunks"]:
                chunk_text = " ".join(sentence_chunk).replace("  ", " ").strip()
                chunk_text = re.sub(r'\.([A-Z])', r'. \1', chunk_text)
                all_chunks.append((file_name, item["page_number"], chunk_text))

        # Check for duplicates before embedding
        unique_chunks = []
        for chunk in all_chunks:
            cursor.execute('''
                SELECT 1 FROM embeddings WHERE file_name = ? AND page_number = ? AND chunk_text = ?
            ''', (chunk[0], chunk[1], chunk[2]))
            if cursor.fetchone() is None:
                unique_chunks.append(chunk)

        # Batch process embeddings
        for i in range(0, len(unique_chunks), BATCH_SIZE):
            batch = unique_chunks[i:i + BATCH_SIZE]
            texts = [chunk[2] for chunk in batch]
            embeddings = embedding_model.encode(texts, batch_size=BATCH_SIZE)

            for chunk, embedding in zip(batch, embeddings):
                cursor.execute('''
                    INSERT OR IGNORE INTO embeddings (file_name, page_number, chunk_text, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (chunk[0], chunk[1], chunk[2], embedding.tobytes()))

    conn.commit()
    conn.close()

# Example usage
directory = "data"
db_path = "embeddings.db"
process_pdf_files(directory, db_path)