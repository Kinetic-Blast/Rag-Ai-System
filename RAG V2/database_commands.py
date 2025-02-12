import sqlite3
from datetime import datetime
import numpy as np

def create_database(db_name: str):
    """Create a new SQLite database with Book and Page tables."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the Book table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date_added TEXT NOT NULL,
        file_name TEXT NOT NULL,
        excluded INTEGER DEFAULT 0,  -- Used for excluding books (0 = active, 1 = excluded)
        focused INTEGER DEFAULT 0    -- Used for focusing on specific books (0 = normal, 1 = focused)
    )
    ''')

    # Create the Page table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER NOT NULL,
        file_name TEXT NOT NULL,
        page_number INTEGER NOT NULL,
        page_char_count INTEGER,
        page_word_count FLOAT,
        page_token_count FLOAT,
        text TEXT,
        embedding BLOB,
        FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE
    )
    ''')

    conn.commit()
    conn.close()

def add_book(db_name: str, book_name: str, file_name: str):
    """Add a book to the database, avoiding duplicates."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Check if the book already exists
    cursor.execute("SELECT COUNT(*) FROM books WHERE name = ? AND file_name = ?", (book_name, file_name))
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    date_added = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("INSERT INTO books (name, date_added, file_name) VALUES (?, ?, ?)", (book_name, date_added, file_name))
    conn.commit()
    conn.close()


def get_book(db_name: str, search_value):
    """Get a book by ID, name, or file name."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Determine if search_value is an integer (book ID) or a string (name/file_name)
    if isinstance(search_value, int):
        cursor.execute("SELECT * FROM books WHERE id = ?", (search_value,))
    else:
        cursor.execute("SELECT * FROM books WHERE file_name = ? OR name = ?", (search_value, search_value))

    book = cursor.fetchone()
    conn.close()
    return book  # Returns None if no book found



def add_page(db_name: str, page_data: dict):
    """Add a page to a book in the database, ensuring no duplicates."""
    search_value = page_data.get('file_name')
    book = get_book(db_name, search_value)

    if not book:
        return

    book_id = book[0]
    page_number = page_data.get('page_number')
    file_name = page_data.get('file_name')
    page_char_count = page_data.get('page_char_count')
    page_word_count = page_data.get('page_word_count')
    page_token_count = page_data.get('page_token_count')
    text = page_data.get('text')
    embedding = np.array(page_data.get('embedding'), dtype=np.float32).tobytes()

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pages WHERE book_id = ? AND page_number = ?", (book_id, page_number))
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    cursor.execute('''
    INSERT INTO pages (book_id, file_name, page_number, page_char_count, page_word_count, page_token_count, text, embedding)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (book_id, file_name, page_number, page_char_count, page_word_count, page_token_count, text, embedding))

    conn.commit()
    conn.close()


def remove_book(db_name: str, search_value: str):
    """Remove a book and all associated pages from the database."""
    book = get_book(db_name, search_value)
    if not book:
        return

    book_id = book[0]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM pages WHERE book_id = ?", (book_id,))
    cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))

    conn.commit()

    # Check the free space before vacuuming
    cursor.execute("PRAGMA freelist_count;")
    free_pages = cursor.fetchone()[0]

    # Only VACUUM if a lot of space is freed
    if free_pages > 1000:  # Adjust threshold based on usage
        cursor.execute("VACUUM")

    conn.close()



def remove_page(db_name: str, file_name: str, page_number: int):
    """Remove a specific page from the database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM pages WHERE file_name = ? AND page_number = ?", (file_name, page_number))
    conn.commit()
    conn.close()


def list_books(db_name: str):
    """List all books in the database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, date_added, file_name FROM books")
    books = cursor.fetchall()
    conn.close()

    return books


def list_pages(db_name: str, search_value: str):
    """List all pages for a specific book."""
    book = get_book(db_name, search_value)
    if not book:
        return []

    book_id = book[0]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT page_number, text FROM pages WHERE book_id = ?", (book_id,))
    pages = cursor.fetchall()
    conn.close()

    return pages


def get_vectors(db_name: str, file_name: str, page_number: int):
    """Retrieve the vector embedding for a specific page."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT embedding FROM pages WHERE file_name = ? AND page_number = ?", (file_name, page_number))
    result = cursor.fetchone()
    conn.close()

    if result:
        return np.frombuffer(result[0], dtype=np.float32)  # Convert from BLOB to NumPy array
    return None


def exclude_book(db_name: str, search_value: str):
    """Mark a book as excluded without deleting it."""
    book = get_book(db_name, search_value)
    if not book:
        return

    book_id = book[0]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("UPDATE books SET excluded = 1 WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()

def include_book(db_name: str, search_value: str):
    """Mark a book as excluded without deleting it."""
    book = get_book(db_name, search_value)
    if not book:
        return

    book_id = book[0]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("UPDATE books SET excluded = 0 WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()


def focus_book(db_name: str, search_value: str):
    """Mark a book as a focus priority."""
    book = get_book(db_name, search_value)
    if not book:
        return

    book_id = book[0]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("UPDATE books SET focused = 1 WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()


def un_focus_book(db_name: str, search_value: str):
    """remove focus priority."""
    book = get_book(db_name, search_value)
    if not book:
        return

    book_id = book[0]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("UPDATE books SET focused = 0 WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()
