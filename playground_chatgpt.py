import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from langchain.llms import Ollama
import anthropic

# Title of the Streamlit app
st.title("Just practicing Streamlit Code")

# Fetch the API keys from streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
llama_api_key = st.secrets["llama_api_key"]
claude_api_key = st.secrets["claude_api_key"]

# Set API keys for OpenAI and Anthropic
openai.api_key = openai_api_key

# SIDEBAR ELEMENTS
st.sidebar.title("Summarization Options")

# SUMMARIZATION
# Model options for summarization
selected_llm_for_summarization = st.sidebar.selectbox(
    "Choose the model",
    (
        "OpenAI: gpt-4o-mini",
        "OpenAI: gpt-4o (Advanced)",
        "LLaMa: llama3.1-8b",
        "LLaMa: llama3.1-405b (Advanced)",
        "Claude: claude-3-haiku-20240307",
        "Claude: claude-3-5-sonnet-20240620 (Advanced)",
    ),
)

if selected_llm_for_summarization == "OpenAI: gpt-4o-mini":
    model_to_use_for_summarization = "gpt-3.5-turbo"
elif selected_llm_for_summarization == "OpenAI: gpt-4o (Advanced)":
    model_to_use_for_summarization = "gpt-4"
elif selected_llm_for_summarization == "LLaMa: llama3.1-8b":
    model_to_use_for_summarization = "llama2-7b"
elif selected_llm_for_summarization == "LLaMa: llama3.1-405b (Advanced)":
    model_to_use_for_summarization = "llama2-70b"
elif selected_llm_for_summarization == "Claude: claude-3-haiku-20240307":
    model_to_use_for_summarization = "claude-instant"
elif (
    selected_llm_for_summarization
    == "Claude: claude-3-5-sonnet-20240620 (Advanced)"
):
    model_to_use_for_summarization = "claude-2"
else:
    model_to_use_for_summarization = None

# Summary format options
option_1 = "Summarize the document in 100 words"
option_2 = "Summarize the document in 2 connecting paragraphs"
option_3 = "Summarize the document in 5 bullet points"
summary_options = st.sidebar.radio(
    "Select a format for summarizing the document:", (option_1, option_2, option_3)
)

# Function to extract text content from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Handle HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator="\n")
    except requests.RequestException as e:
        st.error(f"Failed to retrieve URL: {url}. Error: {e}")
        return None


# Inputs for URLs
url1 = st.text_input("First URL:")
url2 = st.text_input("Second URL:")

# Function to generate summary
def generate_summary(text, instruction, model_to_use):
    if model_to_use in ["gpt-3.5-turbo", "gpt-4"]:
        return summarize_with_openai(text, instruction, model_to_use)
    elif model_to_use.startswith("llama"):
        return summarize_with_llama(text, instruction, model_to_use)
    elif model_to_use.startswith("claude"):
        return summarize_with_claude(text, instruction, model_to_use)
    else:
        st.error("Model not supported.")
        return None


def summarize_with_openai(text, instruction, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
        {"role": "user", "content": f"{instruction}\n\n{text}"},
    ]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, max_tokens=500
    )
    summary = response["choices"][0]["message"]["content"]
    return summary


def summarize_with_llama(text, instruction, model):
    llm = Ollama(model=model)
    prompt = f"{instruction}\n\n{text}"
    response = llm(prompt)
    return response


def summarize_with_claude(text, instruction, model):
    client = anthropic.Client(api_key=claude_api_key)
    prompt = f"{anthropic.HUMAN_PROMPT} {instruction}\n\n{text} {anthropic.AI_PROMPT}"
    response = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model,
        max_tokens_to_sample=500,
    )
    return response["completion"]


# Add a button to generate summary
if st.button("Generate Summary"):
    text1 = extract_text_from_url(url1)
    text2 = extract_text_from_url(url2)

    if text1 is None or text2 is None:
        st.error("Failed to extract text from one or both URLs.")
    else:
        if summary_options == option_1:
            instruction = "Summarize the following document in 100 words."
        elif summary_options == option_2:
            instruction = "Summarize the following document in 2 connecting paragraphs."
        elif summary_options == option_3:
            instruction = "Summarize the following document in 5 bullet points."
        else:
            instruction = "Summarize the following document."

        # Generate summary for text1
        summary1 = generate_summary(text1, instruction, model_to_use_for_summarization)

        # Generate summary for text2
        summary2 = generate_summary(text2, instruction, model_to_use_for_summarization)

        # Display the summaries
        st.write("### Summary of First URL:")
        st.write(summary1)

        st.write("### Summary of Second URL:")
        st.write(summary2)

# CHAT BOT
st.sidebar.title("Chat Bot Options")

# Conversation behavior options
behavior = st.sidebar.radio(
    "Conversation behavior:",
    (
        "Keep last 5 questions",
        "Summarize after 5 interactions",
        "Limit by token size (5000 tokens)",
    ),
)

# Session state for chatbot memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": "Your output should end with 'DO YOU WANT MORE INFO?' case sensitive unless you get a 'no' as an input. In which case, you should go back to asking 'How can I help you?' Make sure your answers are simple enough for a 10-year-old to understand.",
        },
        {"role": "assistant", "content": "How can I help you?"},
    ]

# Model options for CHATBOT
selected_llm_for_chatbot = st.sidebar.selectbox(
    "Choose the model for Chatbot",
    (
        "OpenAI: gpt-4o-mini",
        "OpenAI: gpt-4o (Advanced)",
        "LLaMa: llama3.1-8b",
        "LLaMa: llama3.1-405b (Advanced)",
        "Claude: claude-3-haiku-20240307",
        "Claude: claude-3-5-sonnet-20240620 (Advanced)",
    ),
)

if selected_llm_for_chatbot == "OpenAI: gpt-4o-mini":
    model_to_use_for_chatbot = "gpt-3.5-turbo"
elif selected_llm_for_chatbot == "OpenAI: gpt-4o (Advanced)":
    model_to_use_for_chatbot = "gpt-4"
elif selected_llm_for_chatbot == "LLaMa: llama3.1-8b":
    model_to_use_for_chatbot = "llama2-7b"
elif selected_llm_for_chatbot == "LLaMa: llama3.1-405b (Advanced)":
    model_to_use_for_chatbot = "llama2-70b"
elif selected_llm_for_chatbot == "Claude: claude-3-haiku-20240307":
    model_to_use_for_chatbot = "claude-instant"
elif (
    selected_llm_for_chatbot == "Claude: claude-3-5-sonnet-20240620 (Advanced)"
):
    model_to_use_for_chatbot = "claude-2"
else:
    model_to_use_for_chatbot = None

# Function to manage conversation memory
def manage_memory(messages, behavior):
    if behavior == "Keep last 5 questions":
        return messages[-10:]  # Keeping the last 5 user-assistant pairs
    elif behavior == "Summarize after 5 interactions":
        if len(messages) > 11:  # If more than 5 pairs (10 messages), summarize
            document = "\n".join(
                [msg["content"] for msg in messages if msg["role"] == "user"]
            )
            instruction = "Summarize this conversation."
            summary = generate_summary(
                document, instruction, model_to_use_for_chatbot
            )
            st.write("### Conversation Summary")
            st.write(summary)
            return [{"role": "assistant", "content": summary}]  # Store only the summary
        else:
            return messages
    elif behavior == "Limit by token size (5000 tokens)":
        token_count = sum([len(msg["content"]) for msg in messages])  # Rough estimation
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

    # Function to get chatbot response
    def get_chatbot_response(messages, model_to_use):
        if model_to_use in ["gpt-3.5-turbo", "gpt-4"]:
            return chatbot_response_openai(messages, model_to_use)
        elif model_to_use.startswith("llama"):
            return chatbot_response_llama(messages, model_to_use)
        elif model_to_use.startswith("claude"):
            return chatbot_response_claude(messages, model_to_use)
        else:
            st.error("Model not supported.")
            return None

    def chatbot_response_openai(messages, model):
        response = openai.ChatCompletion.create(
            model=model, messages=messages
        )
        assistant_message = response["choices"][0]["message"]["content"]
        return assistant_message

    def chatbot_response_llama(messages, model):
        llm = Ollama(model=model)
        # Convert messages into a single prompt
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n"
        prompt += "Assistant:"
        response = llm(prompt)
        return response

    def chatbot_response_claude(messages, model):
        client = anthropic.Client(api_key=claude_api_key)
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"{anthropic.HUMAN_PROMPT} {message['content']} {anthropic.AI_PROMPT}"
            elif message["role"] == "user":
                prompt += f"{anthropic.HUMAN_PROMPT} {message['content']} {anthropic.AI_PROMPT}"
            elif message["role"] == "assistant":
                prompt += f"{message['content']}"
        response = client.completion(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model,
            max_tokens_to_sample=500,
        )
        return response["completion"]

    # Get assistant's response
    assistant_message = get_chatbot_response(
        st.session_state["messages"], model_to_use_for_chatbot
    )

    # Append the assistant's response to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
