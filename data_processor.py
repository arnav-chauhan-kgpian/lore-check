import pathway as pw
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

# Initialize model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@pw.udf
def embed_text(text: str) -> list[float]:
    """Wraps HF model to embed text chunks."""
    return embedding_model.encode(text).tolist()

def chunk_text_python(text, chunk_size=500):
    """Simple splitter helper."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

@pw.udf
def chunk_text_udf(text: str) -> list[str]:
    return chunk_text_python(text)

class BookIndexer:
    def __init__(self, books_dir):
        self.books_dir = books_dir
        self.df = None # This will hold our indexed data (Chunks + Vectors)

    def build_index(self):
        """
        Uses Pathway to read, chunk, and embed the books.
        """
        print("--- Pathway: Ingesting Books ---")
        
        # 1. Pathway Connector: Read files from 'Books' folder
        # 'path' metadata helps us link to book_name in CSV
        files = pw.io.fs.read(
            self.books_dir,
            format="binary",
            mode="static",
            with_metadata=True
        )

        # 2. Decode text
        documents = files.select(
            text=pw.this.data.dt.to_string(),
            path=pw.this._metadata.path
        )

        # 3. Clean filename to match 'book_name' in CSV
        # Assuming book_name in CSV is "book.txt" or "book"
        # We extract the filename from the full path
        documents = documents.select(
            pw.this.text,
            filename=pw.this.path.dt.os_path_basename() 
        )

        # 4. Chunking (Flattening the list of chunks)
        chunks = documents.select(
            chunk=chunk_text_udf(pw.this.text),
            pw.this.filename
        ).flatten(pw.this.chunk)

        # 5. Embedding
        embedded_chunks = chunks.select(
            pw.this.filename,
            pw.this.chunk,
            vector=embed_text(pw.this.chunk)
        )

        # 6. Materialize to Pandas for the Query Loop
        # In a full app, we'd use pw.io.csv.write or a server, 
        # but for a static hackathon submission, this is most robust.
        print("Computing embeddings... (this may take a moment)")
        self.df = pw.debug.table_to_pandas(embedded_chunks)
        print(f"Index Built! Total Chunks: {len(self.df)}")
        
    def search(self, query, book_name, k=5):
        """
        Retrieves relevant chunks, strictly filtering by the requested book.
        """
        if self.df is None:
            raise ValueError("Index not built. Call build_index() first.")

        # 1. Filter: Only look at the specific book
        # We try strict match, then loose match (in case csv says "1984" and file is "1984.txt")
        book_df = self.df[self.df['filename'] == book_name]
        if book_df.empty:
            # Try appending .txt if missing
            book_df = self.df[self.df['filename'] == f"{book_name}.txt"]
        
        if book_df.empty:
            print(f"Warning: Book '{book_name}' not found in index.")
            return []

        # 2. Vector Search (Manual Cosine Similarity on the filtered subset)
        # (Pathway handles this in streaming, we do it here for the specific CSV row logic)
        query_vec = embedding_model.encode(query)
        
        # Convert list-vectors in DF to matrix
        import numpy as np
        vectors = np.vstack(book_df['vector'].values)
        
        scores = np.dot(vectors, query_vec)
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = book_df.iloc[top_indices]['chunk'].tolist()
        return results