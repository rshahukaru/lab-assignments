# import streamlit as st
# from openai import OpenAI

# st.title("Just practicing Streamlit Code")

# # 1. Fetching the openai_api_key from streamlit
# openai_api_key = st.secrets["openai_api_key"]


# # 2. 
# if not openai_api_key: 
#     # This code block of if statement is for the cases when the openai_api_key is not defined
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
#     openai_api_key = st.text_input("Enter your OpenAI API key")
# else:
#     # This else part of the code is executed when the openai_api_key exists
    
#     # This is where the logic starts from
#     # 2.1. We create a variable called client, as an instance for interacting with the OpenAI API using the provided key
#     client = OpenAI(api_key=openai_api_key)

#     # Commenting the two lines of code below because they don't serve any purpose in this lab
#     # Let the user upload a file via ⁠ st.file_uploader ⁠.
#     uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

#     # 2.2. Building a Sidebar and its elements

#     # Sidebar Title
#     st.sidebar.title("Options")
    
#     # Sidebar Model Options
#     openAI_model = st.sidebar.selectbox("Choose the model", ("mini", "regular"))
#     model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

#     # Sidebar Summary Options
#     option_1 = "Summarize the document in 100 words"
#     option_2 = "Summarize the document in 2 connecting paragraphs"
#     option_3 = "Summarize the document in 5 bullet points"

#     summary_options = st.sidebar.radio("Select a format for summarizing the document:",
#                                        (option_1, option_2, option_3))
    
#     # 2.3. Writing logic to pass the document along with specific set instructions to the model
#     if uploaded_file:
#         # Processing the uploaded file and storing it in the variable document
#         document = uploaded_file.read().decode() # We use the .decode() method to make the text more readable

#         # Instruction based on the user's selection on the sidebar menu
#         instruction = summary_options.lower()

#         # IMPORTANT
#         # Preparing the messages to pass to the LLM
#         # NOTE: This is the exact format at the time of writing this code that we're supposed to follow before passing it to the model
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"Here's a document: \n{document} \n\n--\n\n Here's some instruction(s): \n{instruction}"
#             }
#         ]

#         # IMPORTANT
#         # THE CODE BELOW IS WHERE WE ACTUALLY PASS OUR INPUT TO THE MODEL AND STORE THE MODEL's OUTPUT INSIDE THE stream VARIABLE
#         stream = client.chat.completions.create(
#             model=model_to_use,
#             messages=messages,
#             stream=True # This stream argument ensures that the output is streamed rather than make the user wait for the output
#         )

#         # IMPORTANT
#         # THE MODEL's STORED OUTPUT IS THEN STREAMED USING THE CODE BELOW
#         st.write_stream(stream)


#     # LAB-03
#     # Setting up the session state to hold chatbot messages



#     hidden_instruction = "Your output should end with 'DO YOU WANT MORE INFO?' unless you get a 'no' as an input. In which case, you should go back to asking 'How can I help you?' Make sure your answers are simple enough for a 10-year-old to understand."
#     # Hidden instructions for the chatbot - Note that the role this time is "system" rather than user or assistant

#     st.session_state["messages"] = [
#         {
#             "role": "assistant",
#             "content": "How can I help you?"
#         },
#         {
#             "role": "system",
#             "content": hidden_instruction
#         }
#     ]


#     # st.write(st.session_state) # checking what the st.session_state contains
#     st.write("### Chatbot")

#     for msg in st.session_state.messages:
#         # Skip system messages from being displayed
#         if msg["role"] == "system":
#             continue

#         # st.chat_message specifies the role of the message
#         chat_msg = st.chat_message(msg["role"])
#         chat_msg.write(msg["content"])


#     # Capturing the user input for the chatbot
#     # The walrus operator (:=) assigns the user's input from st.chat_input("...") to the variable 'prompt'.
#     # The if block runs only if the user provides input (i.e., 'prompt' is not empty).
#     if prompt := st.chat_input("Ask the chatbot a question or interact:"):

#         # Append the user's message to the session state 'messages' list with role and content.
#         st.session_state.messages.append({
#             "role": "user",
#             "content": prompt})

#         # Let's display the user's input in the chat
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Let's now pass the prompt along with the instructions to the model
#         stream = client.chat.completions.create(
#             model=model_to_use,
#             messages=st.session_state["messages"],
#             stream=True # This stream argument ensures that the output is streamed rather than make the user wait for the output
#         )

#         # Stream the assistant's response
#         with st.chat_message("assistant"):
#             response = st.write_stream(stream)

#         # Append the assistant's response to the session state
#         st.session_state.messages.append(
#             {
#                 "role": "assistant",
#                 "content": response
#             }
#         )

# st.write(st.session_state.messages)





########################################################################################################
############################################### NEW CODE ###############################################





import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from langchain_ollama import OllamaLLM

# Title of the Streamlit app
st.title("Just practicing Streamlit Code")

# Fetch the API keys from streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
llama_api_key  = st.secrets["llama_api_key"]
claude_api_key = st.secrets["claude_api_key"]

# SIDEBAR ELEMENTS

st.sidebar.title("Summarization Options")

# SUMMARIZATION
# Model options for summarization
selected_llm_for_summarization = st.sidebar.selectbox("Choose the model", ("OpenAI: gpt-4o-mini", "OpenAI: gpt-4o (Advanced)",
                                                                            "LLaMa: llama3.1-8b", "LLaMa: llama3.1-405b (Advanced)",
                                                                            "Claude: claude-3-haiku-20240307", "Claude: claude-3-5-sonnet-20240620 (Advanced)"))

if selected_llm_for_summarization == "OpenAI: gpt-4o-mini":
    model_to_use_for_summarization = "gpt-4o-mini"

elif selected_llm_for_summarization == "OpenAI: gpt-4o (Advanced)":
    model_to_use_for_summarization = "gpt-4o"

elif selected_llm_for_summarization == "LLaMa: llama3.1-8b":
    model_to_use_for_summarization = "llama3.1-8b"

elif selected_llm_for_summarization == "LLaMa: llama3.1-405b (Advanced)":
    model_to_use_for_summarization = "llama3.1-405b"

elif selected_llm_for_summarization == "Claude: claude-3-haiku-20240307)":
    model_to_use_for_summarization = "claude-3-haiku-20240307"

elif selected_llm_for_summarization == "Claude: claude-3-5-sonnet-20240620 (Advanced)":
    model_to_use_for_summarization = "claude-3-5-sonnet-20240620"

else:
    pass

# SUMMARIZATION
# Summary format options
option_1 = "Summarize the document in 100 words"
option_2 = "Summarize the document in 2 connecting paragraphs"
option_3 = "Summarize the document in 5 bullet points"
summary_options = st.sidebar.radio("Select a format for summarizing the document:",
                                    (option_1, option_2, option_3))



# Function to extract text content from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Handle HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Failed to retrieve URL: {url}. Error: {e}")
        return None
    
# Input for URL
input_url = st.text_input("URL Input:")



# CHAT BOT
st.sidebar.title("Chat Bot Options")

# Conversation behavior options
behavior = st.sidebar.radio("Conversation behavior:", ("Keep last 5 questions", "Summarize after 5 interactions", "Limit by token size (5000 tokens)"))

# Session state for chatbot memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "system", "content": "Your output should end with 'DO YOU WANT MORE INFO?' case sensitive unless you get a 'no' as an input. In which case, you should go back to asking 'How can I help you?' Make sure your answers are simple enough for a 10-year-old to understand."}
    ]




################################### CHATBOT ###################################

# Model options for CHATBOT
selected_llm_for_chatbot = st.sidebar.selectbox("Choose the model for Chatbot", ("OpenAI: gpt-4o-mini", "OpenAI: gpt-4o (Advanced)",
                                                                            "LLaMa: llama3.1-8b", "LLaMa: llama3.1-405b (Advanced)",
                                                                            "Claude: claude-3-haiku-20240307", "Claude: claude-3-5-sonnet-20240620 (Advanced)"))

if selected_llm_for_chatbot == "OpenAI: gpt-4o-mini":
    model_to_use_for_chatbot = "gpt-4o-mini"

elif selected_llm_for_chatbot == "OpenAI: gpt-4o (Advanced)":
    model_to_use_for_chatbot = "gpt-4o"

elif selected_llm_for_chatbot == "LLaMa: llama3.1-8b":
    model_to_use_for_chatbot = "llama3.1-8b"

elif selected_llm_for_chatbot == "LLaMa: llama3.1-405b (Advanced)":
    model_to_use_for_chatbot = "llama3.1-405b"

elif selected_llm_for_chatbot == "Claude: claude-3-haiku-20240307)":
    model_to_use_for_chatbot = "claude-3-haiku-20240307"

elif selected_llm_for_chatbot == "Claude: claude-3-5-sonnet-20240620 (Advanced)":
    model_to_use_for_chatbot = "claude-3-5-sonnet-20240620"

else:
    pass


########################## OpenAI CHATBOT ##########################

client = OpenAI(api_key=openai_api_key)

# Display the chatbot interface
st.write("### Chatbot")

# Implementing conversation buffer, summary, and token size limit
def manage_memory(messages, behavior):

    if behavior == "Keep last 5 questions":
        return messages[-10:]  # Keeping the last 5 user-assistant pairs
    
    elif behavior == "Summarize after 5 interactions":
        if len(messages) > 11:  # If more than 5 pairs (10 messages), summarize
            document = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
            summary_instruction = "Summarize this conversation."
            summary_messages = [
                {"role": "user", "content": f"Here's a conversation: \n{document} \n\nSummarize it: {summary_instruction}"}
            ]
            summary = client.chat.completions.create(model=model_to_use_for_chatbot, messages=summary_messages)
            st.write("### Conversation Summary")
            st.write(summary)
            return [{"role": "assistant", "content": summary}]  # Store only the summary
        else:
            return messages
    
    elif behavior == "Limit by token size (5000 tokens)":
        token_count = sum([len(msg["content"]) for msg in messages])  # Rough estimation by character count
        
        while token_count > 5000:
            messages.pop(0)  # Remove oldest messages until under the token limit
            token_count = sum([len(msg["content"]) for msg in messages])
        return messages

# Manage conversation memory
st.session_state.messages = manage_memory(st.session_state.messages, behavior)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":  # Skip the system messages
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

# Capturing the user input for the chatbot
if prompt := st.chat_input("Ask the chatbot a question or interact:"):

    # Append the user's message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's input in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass the prompt to the OpenAI API along with session messages
    stream = client.chat.completions.create(
        model=model_to_use_for_chatbot,
        messages=st.session_state["messages"],
        stream=True  # Streaming the response from the model
    )

    # Stream the assistant's response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Append the assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})


################################################################################################################




########################## LLaMa - langchain_ollama CHATBOT ##########################

















########################## Claude CHATBOT ##########################




