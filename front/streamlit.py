import uuid
import streamlit as st
import requests

API_BASE_URL = "https://rag-learn-dh29.vercel.app"

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="GenAI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 GenAI Assistant")
st.caption("Powered by FastAPI")

# -----------------------------
# Session state
# -----------------------------
# Keep the thread_id in the URL so a page reload resumes the same
# conversation instead of starting a new, blank one.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.query_params.get("thread_id") or str(uuid.uuid4())
    st.query_params["thread_id"] = st.session_state.thread_id

# Restore saved history from the backend once per session.
if "messages" not in st.session_state:
    try:
        response = requests.get(
            f"{API_BASE_URL}/threads/{st.session_state.thread_id}/messages",
            timeout=30,
        )
        response.raise_for_status()
        st.session_state.messages = response.json()["messages"]
    except Exception:
        st.session_state.messages = []

def switch_thread(thread_id: str, messages: list):
    st.session_state.thread_id = thread_id
    st.query_params["thread_id"] = thread_id
    st.session_state.messages = messages
    st.rerun()

# -----------------------------
# Sidebar: thread history
# -----------------------------
with st.sidebar:
    st.header("Chats")

    if st.button("🧹 New chat", use_container_width=True):
        switch_thread(str(uuid.uuid4()), [])

    st.divider()

    try:
        response = requests.get(f"{API_BASE_URL}/threads", timeout=30)
        response.raise_for_status()
        threads = response.json()["threads"]
    except Exception:
        threads = []

    for thread in threads:
        label = thread["preview"] or "(empty conversation)"
        is_current = thread["thread_id"] == st.session_state.thread_id
        if st.button(
            f"{'💬 ' if is_current else ''}{label}",
            key=f"thread-{thread['thread_id']}",
            use_container_width=True,
            disabled=is_current,
        ):
            try:
                history_response = requests.get(
                    f"{API_BASE_URL}/threads/{thread['thread_id']}/messages",
                    timeout=30,
                )
                history_response.raise_for_status()
                messages = history_response.json()["messages"]
            except Exception:
                messages = []
            switch_thread(thread["thread_id"], messages)

# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input
# -----------------------------
if prompt := st.chat_input("Ask something..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "question": prompt,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=600
                )

                response.raise_for_status()

                data = response.json()

                # Change this depending on your API response
                answer = data["response"]

            except requests.exceptions.ConnectionError:
                answer = "❌ Could not connect to FastAPI."

            except requests.exceptions.Timeout:
                answer = "⏱️ Request timed out."

            except requests.exceptions.HTTPError as e:
                answer = f"❌ API error: {e}"

            except Exception as e:
                answer = f"❌ Error: {e}"

        st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })