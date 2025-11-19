import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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


# ---------------- Caching ----------------
@st.cache_resource
def load_vectorstore():
    """Cache embeddings + vectorstore so they don't rebuild every reload"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_DIR):
        return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)

    # Build fresh if not exists
    if CV_PATH.lower().endswith(".pdf"):
        loader = PyPDFLoader(CV_PATH)
    else:
        loader = TextLoader(CV_PATH, encoding="utf8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(chunks, embeddings, persist_directory=INDEX_DIR)
    return vs


# ---------------- Load once ----------------
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
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
    return_source_documents=False
)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Custom CSS (YOUR ORIGINAL CSS) ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif") no-repeat center center fixed;
        background-size: cover;
        color: #EAF2FF;
        min-height: 100vh;
    }

    .user-bubble, .bot-bubble {
        border-radius: 14px !important;
        padding: 10px 14px !important;
        margin: 6px 0 !important;
        max-width: 75% !important;
        text-align: left !important;
        animation: fadeIn 0.3s ease-in-out;
    }
    .user-bubble {
        background: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    .bot-bubble {
        background: rgba(0, 0, 0, 0.6) !important;
        color: #FFFFFF !important;
    }

    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 12px;
        flex-shrink: 0;
        border: 2px solid black;
        box-shadow: 0 3px 8px rgba(0,0,0,0.4);
    }
    .chat-msg {
        flex: 1;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .stButton>button {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid black!important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }
    .stButton>button:hover {
        border: 1px solid #FFFFFF !important;
        cursor: pointer !important;
    }

    .stChatInput input {
        background-color: black !important;
        border: 2px solid black !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    .stChatInput input:focus {
        border: 2px solid black !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- UI ----------------
st.title("✨ Ask Ayesha's AI Career Bot")
st.write("Ask me anything about **Ayesha's education, skills, and projects** — answers come only from her CV.")

BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

# ---------------- Suggestions at TOP ----------------
if len(st.session_state.messages) == 0:
    st.markdown("#### 🔎 Try asking me:")
    cols = st.columns(3)
    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships does Ayesha have?",
        "What are Ayesha's top technical skills?"
    ]
    for i, text in enumerate(suggestions):
        if cols[i].button(text, key=f"sugg_{i}"):
            st.session_state.messages.append({"role": "user", "content": text})
            with st.spinner("Thinking..."):
                result = qa_chain({"question": text})
            answer = result["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
# ---------------- Chat Container ----------------
chat_container = st.container()  # fixed container for chat messages

# ---------------- Chat Input ----------------
if prompt := st.chat_input("Type your question about Ayesha's CV..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": prompt})
        answer = result["answer"]

    # Append bot message
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- Display Chat History ----------------
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        msg_id = f"msg-{i}"
        is_latest_bot = i == len(st.session_state.messages) - 1 and msg["role"] == "assistant"
        html_id = "latest-bot-msg" if is_latest_bot else msg_id

        if msg["role"] == "assistant":
            st.markdown(
                f"""
                <div class="chat-row" id="{html_id}">
                    <img src="{BOT_AVATAR}" class="chat-avatar">
                    <div class="chat-msg"><div class="bot-bubble">{msg['content']}</div></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row" id="{html_id}">
                    <img src="{USER_AVATAR}" class="chat-avatar">
                    <div class="chat-msg"><div class="user-bubble">{msg['content']}</div></div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------- Scroll latest bot message to TOP ----------------
st.markdown(
    """
    <script>
    const latestBot = document.getElementById('latest-bot-msg');
    if(latestBot){
        // Scroll latest bot message to top
        latestBot.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
    </script>
    """,
    unsafe_allow_html=True
)
