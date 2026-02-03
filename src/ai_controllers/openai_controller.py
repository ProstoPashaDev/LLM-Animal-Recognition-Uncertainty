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

    def ask_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response

    def ask_with_image(self, prompt, image_path):
        """
        Sends a text prompt along with an image to a multimodal LLM.
        image_path: path to the local image file
        """
        base64_img = image_to_base64_data_uri(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_img}",},
                ],
                }],
        )
        return response
