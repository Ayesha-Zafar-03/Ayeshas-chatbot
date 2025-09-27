import streamlit as st
import os
from dotenv import load_dotenv

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
BACKGROUND_GIF = "https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif"


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
    vs.persist()
    return vs


# ---------------- Load once ----------------
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model="llama-3.3-70b-versatile",  # ⚡ fast Groq model
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

# ---------------- Custom CSS ----------------
# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    /* Background GIF */
    .stApp {
        background: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif") no-repeat center center fixed;
        background-size: cover;
        color: #EAF2FF;
        min-height: 100vh;
    }

    /* Chat bubbles */
    .user-bubble {
        background: #1E1E1E !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
        color: #FFFFFF !important;
        margin: 6px 0 !important;
        backdrop-filter: blur(10px) !important;
        border: none !important;
        max-width: 75% !important;
        text-align: left !important;
    }
    .bot-bubble {
        background: rgba(0, 0, 0, 0.6) !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
        color: #FFFFFF !important;
        margin: 6px 0 !important;
        backdrop-filter: blur(10px) !important;
        border: none !important;
        max-width: 75% !important;
        text-align: left !important;
    }

    /* Avatars */
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
        border: 2px solid black;   /* White border only */
        box-shadow: 0 3px 8px rgba(0,0,0,0.4);
    }
    .chat-msg {
        flex: 1;
    }

    /* Chat container for better scrolling */
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        scroll-behavior: smooth;
    }

    /* Remove hover effect on suggestion buttons */
    .stButton>button {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid black!important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: none !important; /* Disable transitions */
    }
    .stButton>button:hover {
        background-color: #1E1E1E !important; /* Same as default state */
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        cursor: pointer !important; /* Keep pointer cursor */
    }

    /* Make chat input border black */
    .stChatInput input {
        background-color: black !important;
        border: 2px solid black !important;
        border-radius: 8px !important;
        background-color: black; /* Keep background white for contrast */
        color:#FFFFFF !important; /* Text color */
    }
    .stChatInput input:focus {
        border: 2px solid black !important; /* Ensure black border on focus */
        box-shadow: none !important; /* Remove default focus shadow */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------- UI ----------------
st.title("✨ Ask Ayesha's AI Career Bot")
st.write("Ask me anything about **Ayesha's education, skills, and projects** — answers come only from her CV.")

# Custom avatar URLs
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"  # grey robot
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"  # black person

# Create a container for chat messages
chat_container = st.container()

with chat_container:
    # Display previous messages
    for i, msg in enumerate(st.session_state.messages):
        # Use unique IDs for each message
        msg_id = f"msg-{i}"
        if msg["role"] == "assistant":
            st.markdown(
                f"""
                <div class="chat-row" id="{msg_id}">
                    <img src="{BOT_AVATAR}" class="chat-avatar">
                    <div class="chat-msg"><div class="bot-bubble">{msg['content']}</div></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row" id="{msg_id}">
                    <img src="{USER_AVATAR}" class="chat-avatar">
                    <div class="chat-msg"><div class="user-bubble">{msg['content']}</div></div>
                </div>
                """,
                unsafe_allow_html=True
            )


# Function to handle new messages and scrolling
def add_message_and_scroll(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    msg_id = f"msg-{len(st.session_state.messages) - 1}"

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-row" id="{msg_id}">
                <img src="{USER_AVATAR}" class="chat-avatar">
                <div class="chat-msg"><div class="user-bubble">{content}</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row" id="{msg_id}">
                <img src="{BOT_AVATAR}" class="chat-avatar">
                <div class="chat-msg"><div class="bot-bubble">{content}</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Improved scrolling script
    scroll_js = f"""
    <script>
    setTimeout(function() {{
        const element = document.getElementById('{msg_id}');
        if (element) {{
            element.scrollIntoView({{ 
                behavior: 'smooth', 
                block: 'end',
                inline: 'nearest'
            }});
        }}
    }}, 100);
    </script>
    """
    st.markdown(scroll_js, unsafe_allow_html=True)


# Chat input
if user_q := st.chat_input("Type your question about Ayesha's CV..."):
    # Add user message
    add_message_and_scroll("user", user_q)

    # Bot response
    with st.spinner("Thinking..."):
        result = qa_chain({"question": user_q})
    answer = result["answer"]

    # Add bot response
    add_message_and_scroll("assistant", answer)

# ---------------- Suggestions ----------------
if len(st.session_state.messages) == 0:
    st.markdown("#### 🔎 Try asking me:")
    cols = st.columns(3)
    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships does Ayesha have?",
        "What are Ayesha's top technical skills?"
    ]
    for i, text in enumerate(suggestions):
        if cols[i].button(text):
            # Add user message
            add_message_and_scroll("user", text)

            # Bot response
            with st.spinner("Thinking..."):
                result = qa_chain({"question": text})
            answer = result["answer"]

            # Add bot response
            add_message_and_scroll("assistant", answer)

            # Force rerun to update the display
            st.rerun()