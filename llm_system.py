from dotenv import load_dotenv
import os
from rag_system import RAG_System
import faiss
from groq import Groq
import tempfile


class LLM_System:
    def __init__(self):
        load_dotenv()
        self.Groq_API_Key = os.getenv("Groq_API_Key")
        self.rag_sys=RAG_System()

    def get_llm_response(self, query, top_k=5):
        """Retrieve relevant information from FAISS and get LLM response."""

        temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
        index_path = os.path.join(temp_dir, "faiss_index.index")
        # Embed the query
        query_embedding = self.rag_sys.embed_text(query).reshape(1, -1)
        # Load the FAISS index from file
        faiss_index = faiss.read_index(index_path)

        # Retrieve the top_k nearest neighbors
        distances, indices = faiss_index.search(query_embedding, top_k)
        chunks = self.rag_sys.load_chunks_from_pickle()
        # Get the corresponding chunks from the index
        retrieved_chunks = []
        for idx in indices[0]:
            if idx >= 0:  # Ensure the index is valid
                retrieved_chunks.append(chunks[idx])  # You need to maintain a separate list of chunks

        # Construct the prompt for the LLM
        prompt = f"Based on the following information, {query}\n\n" + "\n\n".join(retrieved_chunks)
        context_info = "\n\n".join(retrieved_chunks)
        prompt = (
        "You are an AI assistant system designed to provide responses solely based on the information from the given and uploaded documents. "
        "If the user's query relates to the content in these documents, provide the relevant data and answer clearly. "
        "If the query is not related to the provided documents, respond with: 'No information found from uploaded documents.'\n\n"
        f"Context: {context_info}\n\n"
        f"User Query: {query}\n\n"
        "Please provide your answer accordingly."
        )

        # Use the Groq API to get the LLM response
        client = Groq(api_key=self.Groq_API_Key)  # Replace with your actual Groq API key
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        return chat_completion.choices[0].message.content
    
    def chat_with_llm(self, query):
        prompt = (
        "You are an AI assistant designed to provide clear and concise answers to user queries. "
        "Based on the query provided below, respond to the user's question accurately and succinctly.\n\n"
        f"User Query: {query}\n\n"
        "Please provide a direct answer to the user's question, ensuring clarity and precision."
        )

        # Use the Groq API to get the LLM response
        client = Groq(api_key=self.Groq_API_Key)  # Replace with your actual Groq API key
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        return chat_completion.choices[0].message.content

