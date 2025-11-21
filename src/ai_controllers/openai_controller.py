import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIController:

    def __init__(self):
        self.model = "gpt-5"
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def ask_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
