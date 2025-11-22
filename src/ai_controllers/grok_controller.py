import os
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system

load_dotenv()

class XAIController:

    def __init__(self):
        key = os.getenv("XAI_API_KEY")
        self.model = "grok-4-1-fast-non-reasoning"
        self.client = Client(
            api_key=key,
            timeout=3600 # Override default timeout with longer timeout for reasoning models
        )

    def ask_grok(self, prompt):
        chat = self.client.chat.create(model=self.model)
        chat.append(user(prompt))
        response = chat.sample()
        return response.content

