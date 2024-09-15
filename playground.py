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

    # 2.2. Building a Sidebar and its elements

    # Sidebar Title
    st.sidebar.title("Options")
    
    # Sidebar Model Options
    openAI_model = st.sidebar.selectbox("Choose the model", ("mini", "regular"))
    model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

    # Sidebar Summary Options
    option_1 = "Summarize the document in 100 words"
    option_2 = "Summarize the document in 2 connecting paragraphs"
    option_3 = "Summarize the document in 5 bullet points"

    summary_options = st.sidebar.radio("Select a format for summarizing the document:",
                                       (option_1, option_2, option_3))
    
    # Commenting the two lines of code below because they don't serve any purpose in this lab
    # # 2.3. Writing logic to pass the document along with specific set instructions to the model
    # if uploaded_file:
    #     # Processing the uploaded file and storing it in the variable document
    #     document = uploaded_file.read().decode() # We use the .decode() method to make the text more readable

    #     # Instruction based on the user's selection on the sidebar menu
    #     instruction = summary_options.lower()

    #     # IMPORTANT
    #     # Preparing the messages to pass to the LLM
    #     # NOTE: This is the exact format at the time of writing this code that we're supposed to follow before passing it to the model
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": f"Here's a document: \n{document} \n\n--\n\n Here's some instruction(s): \n{instruction}"
    #         }
    #     ]

    #     # IMPORTANT
    #     # THE CODE BELOW IS WHERE WE ACTUALLY PASS OUR INPUT TO THE MODEL AND STORE THE MODEL's OUTPUT INSIDE THE stream VARIABLE
    #     stream = client.chat.completions.create(
    #         model=model_to_use,
    #         messages=messages,
    #         stream=True # This stream argument ensures that the output is streamed rather than make the user wait for the output
    #     )

    #     # IMPORTANT
    #     # THE MODEL's STORED OUTPUT IS THEN STREAMED USING THE CODE BELOW
    #     st.write_stream(stream)


    # LAB-03
    # Set up the session state to hold chatbot messages

    st.write(st.session_state)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


    st.write(st.session_state) # checking what the st.session_state contains
    st.write("### Chatbot")

    for msg in st.session_state.messages:
        st.write(msg)
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])
    

    if a := st.chat_input("Tell me a secret"):
        st.write("Your secret's out")

    # # Get user input for the chatbot
    # if prompt := st.chat_input("Ask the chatbot a question or interact:"):
    #     # Append user input to the session state
    #     st.session_state.messages.append({"role": "user", "content": prompt})

        