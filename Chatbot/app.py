from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model


import getpass
import os
import streamlit as st
import re
from gtts import gTTS
import tempfile
from streamlit_chat import message
from streamlit_local_storage import LocalStorage


try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("ENVIRONMENT VARIABLES NOT SET!!!")
    pass

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSMITH API key (optional): "
    )

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        prompt="Enter your OpenAI API key (required if using OpenAI): "
    )

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass(
        prompt="Enter your GROQ API key (required if using GROQ): "
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Your name is Shovon Raul, You can respond in hindi language with english alphabet. You know and can explain details about news and current affiars. You can elaborate things in point by point scientifically and elaborately. PLease respond in hinglish only"),
        ("user", "Question:{question}")
    ]
)

localS = LocalStorage()



# Dictionary of providers and their models
options = {
    "Alibaba_Cloud": ["qwen-qwq-32b"],
    "DeepSeek": ["deepseek-r1-distill-llama-70b"],
    "Google": ["gemma2-9b-it"],
    "Groq": ["compound-beta", "compound-beta-mini"],
    # "Hugging_Face": ["distil-whisper-large-v3-en"],
    "Meta": [
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-guard-3-8b", 
        "llama3-70b-8192", "llama3-8b-8192", 
        "meta-llama/llama-4-maverick-17b-128e-instruct", 
        "meta-llama/llama-4-scout-17b-16e-instruct", 
        "meta-llama/llama-guard-4-12b"
    ],
    # "Mistral_AI": ["mistral-saba-24b"],
    # "Open_AI": ["whisper-large-v3", "whisper-large-v3-turbo"],
    # "Play_AI": ["playai-tts", "playai-tts-arabic"]
}

Chat_History_Options = [
 'qwen-qwq-32b_Chat_History',
 'deepseek-r1-distill-llama-70b_Chat_History',
 'gemma2-9b-it_Chat_History',
 'compound-beta_Chat_History',
 'compound-beta-mini_Chat_History',
 'llama-3.1-8b-instant_Chat_History',
 'llama-3.3-70b-versatile_Chat_History',
 'llama-guard-3-8b_Chat_History',
 'llama3-70b-8192_Chat_History',
 'llama3-8b-8192_Chat_History',
 'meta-llama_llama-4-maverick-17b-128e-instruct_Chat_History',
 'meta-llama_llama-4-scout-17b-16e-instruct_Chat_History',
 'meta-llama_llama-guard-4-12b_Chat_History'
]


# Sidebar as Navbar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Chat History"))

if page == "Home":
    st.title("AI All in ONE")
    # Create two columns
    col1, col2 = st.columns(2)

    # Place dropdowns in each column
    with col1:
        provider = st.selectbox("Choose a Provider", list(options.keys()))

    with col2:
            selected_model = st.selectbox("Choose a Model", options[provider])
            localStorage_itemKey = selected_model+"_Chat_History"

    Current_Model_ChatHistory = localS.getItem(localStorage_itemKey)


    llm = ChatGroq(model = selected_model)

    # 📌 UI input (handled only if not PlayAI)
    if provider != "Play_AI":
        input_text = st.text_input("Ask Shovon Raul...")
    else:
        input_text = st.text_input("Type something to convert to audio...")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        if input_text:
            if provider == "Play_AI":
                # Use LLM to generate a response first
                response_text = chain.invoke({"question": input_text})

                # Clean it (optional)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

                # Use gTTS or your TTS model to convert response to audio
                tts = gTTS(text=cleaned_response, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format="audio/mp3")
                    st.success("🗣️ Response converted to speech!")
                    
                    # Also show the text response
                    st.markdown(f"**Generated Response:** {cleaned_response}")
            else:
                
                # Get AI response
                raw_response = chain.invoke({"question": input_text})

                # Strip <think> tags and contents
                cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

                message(input_text, is_user=True) 
                message(cleaned_response)

                Current_Conversation = local_storage_item = localS.getItem(localStorage_itemKey)
                if Current_Conversation is None:
                    Current_Conversation = []
                Current_Conversation.append({"role": "user", "content": input_text})
                Current_Conversation.append({"role": "ai", "content": cleaned_response})
                localS.setItem(localStorage_itemKey, Current_Conversation)

                # Display cleaned response in a styled container
                # with st.container():
                #     st.markdown(
                #         f"""
                #         <div style="background-color: #172810; padding: 15px; border-radius: 10px; border: 1px solid #cddcec;">
                #             <strong>🤖 Shovon Raul:</strong><br><br>
                #             {cleaned_response}
                #         </div>
                #         """,
                #         unsafe_allow_html=True
                #     )
    except Exception as e:
        st.error(f"Unexpected error: {e}")

elif page == "Chat History":
    st.title("📜 Chat History")

    selected_model_For_History = st.select_slider("Select a Model", options=Chat_History_Options)
    local_storage_item_for_Current_Chat = localS.getItem(selected_model_For_History)
    
    if local_storage_item_for_Current_Chat is not None:
        for msg in local_storage_item_for_Current_Chat:
            is_user = msg["role"] == "user"
            message(msg["content"], is_user=is_user)
    else:
        st.info("No chat history yet.")