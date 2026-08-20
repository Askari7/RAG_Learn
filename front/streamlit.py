import streamlit as st
import requests

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
if "messages" not in st.session_state:
    st.session_state.messages = []

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
                    "http://127.0.0.1:8000/chat",
                    json={
                        "question": prompt
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