from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os,sys
load_dotenv()

class GroqLLM:
    def __init__(self, api_key:str, model:str="llama-3.3-70b-versatile"):
        self.api_key=api_key
        self.model=model
        self.client=None

    def _initialize_client(self):
        self.client=ChatGroq(api_key=self.api_key, model=self.model)
        return self.client