import streamlit as st
from rag_system import RAG_System
from llm_system import LLM_System
import os
import tempfile

rag_sys=RAG_System()
llm_gen=LLM_System()


# Load the AI robot icon image
icon_image = "image.png"  # Replace with the path to the icon image

# Set the page layout
st.set_page_config(layout="wide")

# Display the AI bot icon
st.image(icon_image, width=100)  # Adjust width as needed
st.title("Chat with GenAI")

#st.title("🗨️ Chat with GenAI")
st.write("Ask questions based on the uploaded PDF or just interact with the AI.")

# Left Sidebar: PDF file uploader and options for chat mode
with st.sidebar:
    st.title("📄 PDF Chatbot")
    st.subheader("Upload PDF")
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Create a temporary directory to store the uploaded file
        temp_dir = tempfile.gettempdir()  # Get the temporary directory
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)  # Construct the full path

        # Write the uploaded file to the temporary path
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Use getbuffer() for bytes-like object

        rag_sys.process_docs_store_faiss(temp_file_path)
        st.success("Processed uploaded document successfully...!")
    else:
        st.info("Please upload a PDF to extract text.")
    
    # Chat mode checkboxes
    chat_with_pdf = st.checkbox("Chat with PDF", value=True)
    chat_with_llm = st.checkbox("Chat with LLMs", value=False)
    
    # Logic to ensure only one checkbox is selected at a time
    if chat_with_pdf and chat_with_llm:
        st.warning("Please select only one option at a time.")
        chat_with_pdf = False
        chat_with_llm = False

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input field at the bottom of the chat window
with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", "")
    submit_button = st.form_submit_button("Send")

# Append user's message and LLM's response to chat history
if submit_button and user_input:
    # Add user's input to the session messages
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Handle response based on the chat mode
    if chat_with_pdf and uploaded_file is not None:
        # Simulate a PDF-based response
        llm_response = llm_gen.get_llm_response(user_input)  # Replace with actual logic
        #llm_response = "This is a response based on the PDF content."
    elif chat_with_llm:
        # Simulate an LLM response
        llm_response = llm_gen.chat_with_llm(user_input)
        #llm_response = "This is a response from the LLM."
    else:
        llm_response = "Please upload a PDF or select Chat with LLMs."

    # Append assistant's response to the session messages
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

# Display chat history in a visually appealing way
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**AI:** {message['content']}")