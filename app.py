import os,requests
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

langchain_api_key=os.getenv("GROQ_API_KEY")
print(langchain_api_key)
# llm=ChatGroq(api_key=langchain_api_key, model="llama-3.1-8b-instant")

# response = llm.invoke([HumanMessage(content="Hello!")])
# print(response.content)

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {langchain_api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())