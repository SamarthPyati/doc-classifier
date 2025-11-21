import streamlit as st
import asyncio
import subprocess
import platform
import os
from src.system import RAGSystem
from src.config import RAGConfig
from src.models import Result

# Page configuration
st.set_page_config(
    page_title="DocClassifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-like CSS
st.markdown("""
<style>
    /* General Reset & Colors */
    :root {
        --bg-color: #343541;
        --sidebar-bg: #202123;
        --text-color: #ECECF1;
        --user-msg-bg: #343541;
        --bot-msg-bg: #444654;
        --input-bg: #40414F;
        --border-color: #565869;
    }
    
    /* Main Background */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid #4d4d4f;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    
    /* Chat Messages */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 1.5rem;
        margin: 0;
    }
    
    /* Assistant Message Background */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: var(--bot-msg-bg);
    }
    
    /* User Message Background (Transparent/Default) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: var(--user-msg-bg);
    }

    /* Avatar Styling */
    .stChatMessage .st-emotion-cache-1p1m4ay {
        background-color: #19c37d; /* ChatGPT Green */
        color: white;
    }
    
    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    .stChatInputContainer textarea {
        background-color: var(--input-bg);
        color: white;
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* Sidebar Buttons */
    .stButton button {
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 5px;
        text-align: left;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #2A2B32;
        border-color: #2A2B32;
    }
    
    /* Primary Button (New Chat) */
    div[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: var(--border-color);
        border: 1px solid var(--border-color);
        color: white;
        text-align: center;
    }
    div[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: #40414F;
    }
    
    /* Typography */
    h1, h2, h3, p, div {
        font-family: 'Söhne', 'ui-sans-serif', 'system-ui', -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Noto Sans', sans-serif, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: transparent;
        color: #acacbe;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG System
# NOTE: We do NOT cache the RAGSystem because it holds references to asyncio loops (via gRPC/LLM)
# which become invalid across Streamlit reruns. Re-initializing ensures we use the current loop.
def get_rag_system():
    config = RAGConfig()
    system = RAGSystem(config)
    return system

try:
    rag_system = get_rag_system()
except Exception as e:
    st.error(f"Failed to initialize RAG System: {e}")
    st.stop()

# Helper to open files locally
def open_file(path: str):
    try:
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(path)
        else:                                   # linux variants
            subprocess.call(('xdg-open', path))
    except Exception as e:
        st.error(f"Could not open file: {e}")

# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# We need to initialize the system to get the session store, but we can't await here easily.
# We'll initialize lazily or synchronously if possible, but RAGSystem.initialize is async.
# Workaround: Run initialization in a loop just for setup if needed, OR await it during the first async action.
# However, we need `rag_system.session_store` immediately for the sidebar.
# So we MUST run initialize() now.

try:
    # Get the current loop or create a new one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(rag_system.initialize())
    
except Exception as e:
    st.error(f"Error initializing RAG system: {e}")
    st.stop()

if "session_id" not in st.session_state:
    # Try to load the most recent session
    sessions = rag_system.session_store.list_sessions()
    if sessions:
        last_session_id = sessions[0]
        st.session_state.session_id = last_session_id
        st.session_state.messages = rag_system.get_chat_history(last_session_id)
    else:
        st.session_state.session_id = rag_system.create_new_session()

from src.constants import LLMModel

# ... (imports)

# Sidebar - Configuration & Session Management
with st.sidebar:
    st.title("Settings")
    
    # LLM Selection
    current_model = rag_system.config.LLM.model
    # Find enum member for current model name
    try:
        default_index = list(LLMModel).index(current_model)
    except ValueError:
        default_index = 0
        
    selected_model_name = st.selectbox(
        "Select Model", 
        options=[m.value for m in LLMModel],
        index=default_index
    )
    
    # If model changed, update config and re-initialize
    if selected_model_name != current_model:
        rag_system.config.LLM.model = selected_model_name
        # Re-initialize with new model
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(rag_system.initialize())
            st.success(f"Switched to {selected_model_name}")
        except Exception as e:
            st.error(f"Failed to switch model: {e}")

    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New chat", type="primary", use_container_width=True):
        st.session_state.session_id = rag_system.create_new_session()
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("History")
    
    # List sessions
    sessions = rag_system.session_store.list_sessions()
    
    for sid in sessions:
        title = rag_system.session_store.get_session_title(sid)
        # Truncate title
        display_title = (title[:20] + '..') if len(title) > 20 else title
        
        # Highlight current session
        is_active = sid == st.session_state.session_id
        label = f"👉 {display_title}" if is_active else f"💬 {display_title}"
        
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(label, key=f"sel_{sid}", use_container_width=True):
                # Load the session history
                st.session_state.session_id = sid
                history = rag_system.get_chat_history(sid)
                # Debug: Show what we got
                print(f"DEBUG: Loading session {sid}, got {len(history)} messages")
                for msg in history:
                    print(f"  Role: {msg.get('role')}, Content preview: {msg.get('content', '')[:50]}")
                st.session_state.messages = history
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sid}", help="Delete chat"):
                rag_system.session_store.delete_session(sid)
                # If deleted active session, reset state
                if sid == st.session_state.session_id:
                    del st.session_state.session_id
                    st.session_state.messages = []
                st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear All", use_container_width=True):
        # Logic to clear all? Or just clear current? User asked for delete button on side.
        # Keeping existing clear current logic but renaming for clarity if needed.
        rag_system.clear_chat_history(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
# We don't use st.title to keep it clean like ChatGPT
if not st.session_state.messages:
    st.markdown("<h1 style='text-align: center; color: #ECECF1; margin-top: 20vh;'>DocClassifier</h1>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Display metadata for assistant messages
        if role == "assistant":
            if isinstance(message, dict) and "sources" in message:
                with st.expander("View Sources"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Confidence: {message.get('confidence', 0):.2f}")
                    with col2:
                        st.caption(f"Time: {message.get('processing_time', 0):.2f}s")
                    
                    for source in message.get("sources", []):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.code(source, language=None)
                        with c2:
                            if st.button("Open", key=f"open_{source}_{hash(content)}"):
                                open_file(source)

# Chat Input
if prompt := st.chat_input("Send a message..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Session Title if it's the first message
    if len(st.session_state.messages) <= 2:
        async def generate_title():
            await rag_system.generate_session_title(st.session_state.session_id, prompt)
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(generate_title())
        except:
            pass

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        async def stream_response():
            full_response = ""
            final_result = None
            async for chunk in rag_system.stream_chat(prompt, st.session_state.session_id):
                if isinstance(chunk, Result):
                    final_result = chunk
                else:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response, final_result

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            full_response, final_result = loop.run_until_complete(stream_response())
            
            if final_result:
                msg_data = {
                    "role": "assistant", 
                    "content": full_response,
                    "sources": final_result.sources,
                    "confidence": final_result.confidence,
                    "processing_time": final_result.processing_time
                }
                st.session_state.messages.append(msg_data)
                
                if final_result.sources:
                    with st.expander("View Sources"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Confidence: {final_result.confidence:.2f}")
                        with col2:
                            st.caption(f"Time: {final_result.processing_time:.2f}s")
                        
                        for source in final_result.sources:
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.code(source, language=None)
                            with c2:
                                if st.button("Open", key=f"open_new_{source}"):
                                    open_file(source)
                            
        except Exception as e:
            st.error(f"Error generating response: {e}")
