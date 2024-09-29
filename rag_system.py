import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse 
from llama_index.core import SimpleDirectoryReader
import nest_asyncio  # noqa: E402
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np
import pickle
from dotenv import load_dotenv
import tempfile


class RAG_System:
    def __init__(self):
        load_dotenv()
        self.Llama_api_key = os.getenv("Llama_API_Key")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def document_parser(self, file_path):
        parser = LlamaParse(api_key=self.Llama_api_key, result_type="markdown")
        nest_asyncio.apply()
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
        return documents
    
    def chunk_text_with_splitter(self, text, chunk_size=1000, chunk_overlap=100):
        """Split the text into smaller chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        return chunks
    
    # Initialize FAISS Index
    def create_faiss_index(self, dim):
        """Create a FAISS index for the given dimension."""
        index = faiss.IndexFlatL2(dim)  # Using L2 distance
        return index
    
    # Function to store chunks into a pickle file
    def store_chunks_to_pickle(self, chunks, file_name='text_chunks.pkl'):
        """Append chunks to a pickle file."""
        try:
            # Load existing chunks if file exists
            with open(file_name, 'rb') as file:
                existing_chunks = pickle.load(file)
        except FileNotFoundError:
            # If file doesn't exist, start with an empty list
            existing_chunks = []

        # Append new chunks to existing ones
        existing_chunks.extend(chunks)

        # Store the updated chunks back to the pickle file
        with open(file_name, 'wb') as file:
            pickle.dump(existing_chunks, file)

    def embed_text(self, text):
        """Convert text to embeddings."""
        return self.model.encode(text).astype('float32')  # Ensure embeddings are in float32 format

    # Store embeddings in FAISS
    def store_embeddings(self, index, chunks):
        """Store embeddings in FAISS index."""
        embeddings = []
        for chunk in chunks:
            chunk_embedding = self.embed_text(chunk)
            embeddings.append(chunk_embedding)

        # Convert to a numpy array and add to the FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
        return index
    
    def process_docs_store_faiss(self, file_path):
        temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
        index_path = os.path.join(temp_dir, "faiss_index.index")

        # Process all documents and store their chunks in FAISS
        faiss_index = self.create_faiss_index(dim=384)  # Set dimension to match your model's output size
        documents = self.document_parser(file_path)
        for doc_idx, document in enumerate(documents):
            # Extract text from each document
            doc_text = getattr(document, 'text', "")
            
            if doc_text:  # Only process if the document has text
                # Split the document into chunks
                chunks = self.chunk_text_with_splitter(doc_text, chunk_size=1000, chunk_overlap=100)

                file_name='text_chunks.pkl'
                pkl_path = os.path.join(temp_dir, file_name)
                # Store chunks in pickle file
                self.store_chunks_to_pickle(chunks, pkl_path)

                # Store chunks in FAISS
                faiss_index = self.store_embeddings(faiss_index, chunks)

        # Save the FAISS index to disk
        faiss.write_index(faiss_index, index_path)


    def load_chunks_from_pickle(self,file_name='text_chunks.pkl'):
        temp_dir = tempfile.gettempdir() 
        pkl_path = os.path.join(temp_dir, file_name)
        with open(pkl_path, 'rb') as file:
            chunks = pickle.load(file)
        return chunks


    


    