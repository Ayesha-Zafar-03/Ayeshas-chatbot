import streamlit as st
import os
from dotenv import load_dotenv
import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ---------------- Setup ----------------
st.set_page_config(page_title="Ayesha's Career Chatbot", layout="centered")

# Load API keys
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Add it in your .env file.")
    st.stop()

CV_PATH = "cv.pdf"
INDEX_DIR = "chroma_index"

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .stApp {background-color: #F0F4F8; color: #19183B;}
    .stChatMessage {background-color: #A1C2BD; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: #19183B;}
    .stChatInput {background-color: #E7F2EF; border: none; color: #19183B;}
    .stButton>button {background-color: #A1C2BD; color: #19183B; border: none; padding: 10px 20px; border-radius: 5px;}
    .stButton>button:hover {background-color: #708993; color: #E7F2EF;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Chat with Ayesha — Career Bot")
st.write("Ask me about my education, skills, and projects. Answers come from my CV.")

# ---------------- Helpers ----------------
def load_docs(path):
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf8")
    return loader.load()

def build_vectorstore(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=INDEX_DIR)
    vs.persist()
    return vs

def get_vectorstore(embeddings, rebuild=False):
    # Close previous Chroma connection
    if "vectorstore" in st.session_state:
        try:
            st.session_state.vectorstore.persist()
            st.session_state.vectorstore._client.close()
        except Exception:
            pass

    # Delete index if rebuild requested
    if rebuild and os.path.exists(INDEX_DIR):
        try:
            shutil.rmtree(INDEX_DIR)
        except PermissionError:
            st.warning("⚠️ Cannot delete Chroma index: file is in use. Restart the app to rebuild.")
            return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)

    if os.path.exists(INDEX_DIR):
        return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)
    else:
        docs = load_docs(CV_PATH)
        return build_vectorstore(docs, embeddings)

# ---------------- Load Vectorstore ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = get_vectorstore(embeddings, rebuild=True)  # safely rebuild
st.session_state.vectorstore = vs

# ---------------- RAG Chain ----------------
retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 20})

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Display Chat ----------------
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ---------------- Chat Input ----------------
if user_q := st.chat_input("Type your question about my CV..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_q})
        answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- Suggestion Buttons ----------------
st.markdown("#### 🔎 Quick Questions:")
suggestions = [
    "Tell me about Ayesha’s projects",
    "What internships or experiences does Ayesha have?",
    "What tools does she use for editing?",
    "What events has she attended?"
]

with st.container():
    cols = st.columns(len(suggestions))
    for i, text in enumerate(suggestions):
        if cols[i].button(text):
            st.session_state.messages.append({"role": "user", "content": text})
            with st.chat_message("user"):
                st.markdown(text)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = qa_chain({"question": text})
                answer = result["answer"]
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
