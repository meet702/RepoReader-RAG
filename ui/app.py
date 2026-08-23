import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Software Engineering Agent", layout="wide")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repos" not in st.session_state:
    st.session_state.repos = []

def fetch_repos():
    try:
        response = requests.get(f"{API_URL}/repos")
        if response.status_code == 200:
            st.session_state.repos = response.json().get("repos", [])
    except Exception as e:
        st.error(f"Could not connect to API: {e}")

# Fetch repos on load
if not st.session_state.repos:
    fetch_repos()

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    st.subheader("Ingest Repository")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
    if st.button("Ingest"):
        if repo_url:
            with st.spinner("Ingesting repository... This may take a while."):
                try:
                    res = requests.post(f"{API_URL}/ingest", json={"repo_url": repo_url})
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Success! {data['chunks_created']} chunks, {data['graph_nodes']} nodes.")
                        fetch_repos()
                    else:
                        st.error(f"Error: {res.json().get('detail')}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
        else:
            st.warning("Please enter a URL")
            
    st.subheader("Select Repository")
    if st.session_state.repos:
        selected_repo = st.selectbox("Active Repository", st.session_state.repos)
    else:
        st.info("No repositories ingested yet.")
        selected_repo = None
        
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Main Chat Interface
st.title("AI Software Engineering Agent")

if not selected_repo:
    st.warning("Please select or ingest a repository to start chatting.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_used") and msg["tool_used"] != "None":
                st.caption(f"🔧 used: {msg['tool_used']}")
                
    # Chat Input
    if prompt := st.chat_input("Ask a question about the repository..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "repo_name": selected_repo,
                        "session_id": st.session_state.session_id,
                        "question": prompt
                    }
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        tool_used = data["tool_used"]
                        
                        st.markdown(answer)
                        if tool_used != "None":
                            st.caption(f"🔧 used: {tool_used}")
                            
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "tool_used": tool_used
                        })
                    else:
                        st.error(f"Error: {res.json().get('detail')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
