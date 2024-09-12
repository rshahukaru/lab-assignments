import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
openai_api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

# Title
st.title("📄 LAB 03 - Building Chat Bot")

# Select the model variant
openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

# Initialize session state for messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Function to handle chat input and responses
def handle_chat():
    # Text input for user query
    user_input = st.text_input("Your question:", key="input")

    # Button to submit response
    if st.button("Send"):
        if user_input:
            # Simulating a response for demonstration
            response = f"This is a response to: '{user_input}' using model {model_to_use}"
            # Append user input and response to the chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Clear the text input (This line should be adjusted if causing issues)
            st.session_state['input'] = ''

# Display chat messages
for message in st.session_state.messages:
    st.container().markdown(f"**{message['role'].title()}**: {message['content']}")

# Input and button to handle chat
handle_chat()
