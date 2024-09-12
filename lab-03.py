import streamlit as st
from openai import OpenAI

openai_api_key = st.secrets["openai_api_key"]

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Title
st.title("📄 LAB 03 - Building Chat Bot :blue[(SUID: 226494782)]")


# message = st.chat_message("assistant")
# message.write("Hello Human")

# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following part: {prompt}")

# with st.chat_message("assistant"):
#     response = st.stream(response_generator())
# st.session_state.messages.append({"role": "assistant",
#                                   "content": response})


openAI_model = st.sidebar.selectbox("Which Model?", 
                                    ("mini", "regular"))

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# Create an OpenAI client 
if 'client' not in st.session_state:
    api_key = st.secrets["openai_api_key"]
    st.session_chat["messages"] = \
        [{"role": "assistant","content": "How can I help you?"}]
    
# History 
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])