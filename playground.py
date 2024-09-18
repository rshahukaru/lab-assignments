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

# import streamlit as st
# from openai import OpenAI

# # Title of the Streamlit app
# st.title("Just practicing Streamlit Code")

# # Fetch the OpenAI API key from streamlit secrets
# openai_api_key = st.secrets["openai_api_key"]

# # If no API key is available, ask for it
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
#     openai_api_key = st.text_input("Enter your OpenAI API key")
# else:
#     # Create an OpenAI client
#     client = OpenAI(api_key=openai_api_key)

#     # Sidebar for user to upload document
#     uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

#     # Sidebar elements
#     st.sidebar.title("Options")
#     # Model options
#     openAI_model = st.sidebar.selectbox("Choose the model", ("mini", "regular"))
#     model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

#     # Summary format options
#     option_1 = "Summarize the document in 100 words"
#     option_2 = "Summarize the document in 2 connecting paragraphs"
#     option_3 = "Summarize the document in 5 bullet points"
#     summary_options = st.sidebar.radio("Select a format for summarizing the document:", (option_1, option_2, option_3))

#     # Conversation behavior options
#     behavior = st.sidebar.radio("Conversation behavior:", ("Keep last 5 questions", "Summarize after 5 interactions", "Limit by token size (5000 tokens)"))

#     # Session state for chatbot memory
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [
#             {"role": "assistant", "content": "How can I help you?"},
#             {"role": "system", "content": "Your output should end with 'DO YOU WANT MORE INFO?' unless you get a 'no' as an input. In which case, you should go back to asking 'How can I help you?' Make sure your answers are simple enough for a 10-year-old to understand."}
#         ]
    
#     # Display the chatbot interface
#     st.write("### Chatbot")

#     # Implementing conversation buffer, summary, and token size limit
#     def manage_memory(messages, behavior):

#         if behavior == "Keep last 5 questions":
#             return messages[-10:]  # Keeping the last 5 user-assistant pairs
        
#         elif behavior == "Summarize after 5 interactions":
#             if len(messages) > 10:  # If more than 5 pairs (10 messages), summarize
#                 document = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
#                 summary_instruction = "Summarize this conversation."
#                 summary_messages = [
#                     {"role": "user", "content": f"Here's a conversation: \n{document} \n\nSummarize it: {summary_instruction}"}
#                 ]
#                 summary = client.chat.completions.create(model=model_to_use, messages=summary_messages)
#                 st.write("### Conversation Summary")
#                 st.write(summary)
#                 return [{"role": "assistant", "content": summary}]  # Store only the summary
#             else:
#                 return messages
        
#         elif behavior == "Limit by token size (5000 tokens)":
#             token_count = sum([len(msg["content"]) for msg in messages])  # Rough estimation by character count
            
#             while token_count > 5000:
#                 messages.pop(0)  # Remove oldest messages until under the token limit
#                 token_count = sum([len(msg["content"]) for msg in messages])
#             return messages

#     # Manage conversation memory
#     st.session_state.messages = manage_memory(st.session_state.messages, behavior)

#     # Display chat history
#     for msg in st.session_state.messages:
#         if msg["role"] != "system":  # Skip the system messages
#             chat_msg = st.chat_message(msg["role"])
#             chat_msg.write(msg["content"])

#     # Capturing the user input for the chatbot
#     if prompt := st.chat_input("Ask the chatbot a question or interact:"):

#         # Append the user's message to session state
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Display user's input in the chat
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Pass the prompt to the OpenAI API along with session messages
#         stream = client.chat.completions.create(
#             model=model_to_use,
#             messages=st.session_state["messages"],
#             stream=True  # Streaming the response from the model
#         )

#         # Stream the assistant's response
#         with st.chat_message("assistant"):
#             response = st.write_stream(stream)

#         # Append the assistant's response to session state
#         st.session_state.messages.append({"role": "assistant", "content": response})




########################################################################################################
############################################### NEW CODE ###############################################


# Importing necessary libraries
import streamlit as st  # For building the web application
import openai  # For interacting with OpenAI's API
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT  # For interacting with Claude
from llama_cpp import Llama  # For interacting with Llama models
import os  # For setting environment variables

# Setting the title of the Streamlit app
st.title("Chatbot with Multiple LLMs")

# Fetching API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai_api_key", "")
claude_api_key = st.secrets.get("claude_api_key", "")
llama_api_key = st.secrets.get("llama_api_key", "")

# Setting environment variables for the API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["ANTHROPIC_API_KEY"] = claude_api_key

# Displaying the sidebar for user options
st.sidebar.title("Options")

# Providing a selectbox in the sidebar for the user to choose the model
llm_model = st.sidebar.selectbox(
    "Choose the model",
    ("OpenAI - GPT-3.5", "OpenAI - GPT-4", "Llama", "Claude - Claude 2")
)

# Adding a button to confirm the model change
if st.sidebar.button("Change the chatbot model"):
    # Storing the selected model in the session state when the button is clicked
    st.session_state["selected_model"] = llm_model
    # Resetting the conversation history when the model changes
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Displaying the currently selected model
if "selected_model" in st.session_state:
    # Showing the user which model is currently being used
    st.write(f"**Using model:** {st.session_state['selected_model']}")
else:
    # Informing the user that no model has been selected yet
    st.write("No model selected yet")

# Initializing the conversation messages in session state
if "messages" not in st.session_state:
    # Setting up default messages with an assistant greeting
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Defining a function to get the model response based on the selected LLM
def get_model_response(prompt, messages):
    """
    Selecting the appropriate API client based on the selected model,
    and generating a response from the model.
    """
    selected_model = st.session_state["selected_model"]
    
    # Handling OpenAI models
    if "OpenAI" in selected_model:
        # Determining which OpenAI model to use based on the selection
        openai_model = "gpt-3.5-turbo" if "3.5" in selected_model else "gpt-4"
        # Setting the OpenAI API key
        openai.api_key = openai_api_key
        # Generating the response from the OpenAI model
        response = openai.ChatCompletion.create(
            model=openai_model,
            messages=messages,
            stream=True
        )
        # Returning the response stream
        return response

    # Handling Llama model
    elif "Llama" in selected_model:
        # Loading the Llama model; assuming model weights are available locally
        # Replace 'path_to_llama_model' with the actual path to your Llama model
        llm = Llama(model_path="path_to_llama_model")
        # Preparing the conversation history as a single prompt
        conversation = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation += f"Assistant: {msg['content']}\n"
        # Generating the response from the Llama model
        output = llm(conversation + f"User: {prompt}\nAssistant:", max_tokens=256, stop=["\nUser:"])
        # Extracting the assistant's reply
        assistant_reply = output['choices'][0]['text'].strip()
        # Returning the assistant's reply
        return assistant_reply

    # Handling Claude model
    elif "Claude" in selected_model:
        # Creating an Anthropic client with the API key
        client = Anthropic(api_key=claude_api_key)
        # Preparing the conversation history for Claude
        conversation = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation += f"{HUMAN_PROMPT} {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation += f"{AI_PROMPT} {msg['content']}\n"

        # Adding the latest user prompt
        conversation += f"{HUMAN_PROMPT} {prompt}\n{AI_PROMPT}"
        # Generating the response from Claude
        response = client.completions.create(
            model="claude-2",
            prompt=conversation,
            max_tokens_to_sample=1000,
            stream=True
        )
        # Returning the response stream
        return response

# Displaying chat history
for msg in st.session_state["messages"]:
    # Creating a chat message with the appropriate role (user or assistant)
    chat_msg = st.chat_message(msg["role"])
    # Writing the content of the message in the chat
    chat_msg.write(msg["content"])

# Capturing user input from the chat input box
if prompt := st.chat_input("Ask the chatbot a question or interact:"):
    # Adding the user's message to the conversation history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Displaying the user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Getting the assistant's response from the selected model
    response = get_model_response(prompt, st.session_state["messages"])
    
    # Displaying the assistant's response in the chat
    with st.chat_message("assistant"):
        # For OpenAI and Claude (streaming responses)
        if hasattr(response, '__iter__') and not isinstance(response, str):
            assistant_response = ""
            for chunk in response:
                if 'choices' in chunk:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        content = delta['content']
                        assistant_response += content
                        st.write(content, end='')
            st.write()
        else:
            # For Llama and non-streaming responses
            st.markdown(response)
            assistant_response = response
        
    # Adding the assistant's response to the conversation history
    st.session_state["messages"].append({"role": "assistant", "content": assistant_response})
