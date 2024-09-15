import streamlit as st
from openai import OpenAI

st.title("Just practicing Streamlit Code")

# 1. Fetching the openai_api_key from streamlit
openai_api_key = st.secrets["openai_api_key"]


# 2. 
if not openai_api_key: 
    # This code block of if statement is for the cases when the openai_api_key is not defined
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    openai_api_key = st.text_input("Enter your OpenAI API key")
else:
    # This else part of the code is executed when the openai_api_key exists

    # This is where the logic starts from
    # 2.1. We create a variable called client, as an instance for interacting with the OpenAI API using the provided key
    client = OpenAI(api_key=openai_api_key)

    # Commenting the two lines of code below because they don't serve any purpose in this lab
    # # Let the user upload a file via ⁠ st.file_uploader ⁠.
    # uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    # 2.2. Creating a Sidebar
    st.sidebar.title("Options")