import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiController:

    def __init__(self):
        self.model = "gemini-3-pro-preview"
        key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)

    def ask_with_image(self, prompt: str, image_path: str):
        img = self.client.files.upload(file=image_path)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[img, prompt],
        )
        return response
