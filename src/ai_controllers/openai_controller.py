import os
from openai import OpenAI
from dotenv import load_dotenv

from src.services.base64_image_encoder import image_to_base64_data_uri

load_dotenv()


class OpenAIController:

    def __init__(self):
        self.model = "gpt-5"
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def ask_gpt(self, prompt: str):
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )
        return response

    def ask_with_image(self, prompt: str, image_path: str):
        base64_img = image_to_base64_data_uri(image_path)

        response = self.client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": base64_img
                    }
                ],
            }]
        )

        return response

    def get_tokens(self, response):
        return response.usage.total_tokens

    def get_text(self, response):
        return response.output_text

    def get_ai_name(self):
        return "ChatGPT"
