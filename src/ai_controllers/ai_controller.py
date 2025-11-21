import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class AIController:

    def __init__(self, model):
        self.models = {"gpt": ("OPENAI_API_KEY", "gpt-5")}
        self.model = self.models[model][1]
        if model == "gpt":
            key = os.getenv(self.models[model][0])
            self.client = OpenAI(api_key=key)

    def ask_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
