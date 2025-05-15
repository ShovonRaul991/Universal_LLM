from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
import getpass

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Your name is Shovon Raul, You can respond in hindi language with english alphabet. You know and can explain details about news and current affiars. You can elaborate things in point by point scientifically and elaborately. PLease respond in hinglish only"),
        ("user", "Question:{question}")
    ]
)

st.title("Shovon Raul CHATBOT")
input_text = st.text_input("Enter your askings...")

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt| llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))