import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiController:

    def __init__(self):
        self.model = "gemini-3-pro-preview"
        key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)

    def ask_with_image(self, prompt: str, image_path: str):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = self.client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                prompt
            ]
        )
        return response

    def get_tokens(self, response):
        return response.usage_metadata.total_token_count

    def get_text(self, response):
        return response.text

    def get_ai_name(self):
        return "Gemini"
