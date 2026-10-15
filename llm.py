# llm.py
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

api_key = dotenv_values("./.env").get("api_key")

def get_llm():
    return ChatOpenAI(api_key=api_key,
                      base_url="https://openrouter.ai/api/v1",
                      model="openrouter/free",
                      temperature=0.8)