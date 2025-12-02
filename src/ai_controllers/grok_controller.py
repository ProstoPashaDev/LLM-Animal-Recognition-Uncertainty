import os

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, image

from src.services.base64_image_encoder import image_to_base64_data_uri

load_dotenv()


class XAIController:

    def __init__(self):
        key = os.getenv("XAI_API_KEY")
        print(key)
        self.model = "grok-4-1-fast-non-reasoning"
        self.client = Client(
            api_key=key,
            timeout=3600  # Override default timeout with longer timeout for reasoning models
        )

    def ask_grok(self, prompt):
        chat = self.client.chat.create(model=self.model)
        chat.append(user(prompt))
        response = chat.sample()
        return response.content

    def ask_grok_with_image(self, prompt: str, image_path: str):
        """
        Send a prompt + an already constructed Image object to Grok.

        Args:
            prompt (str): text prompt
            image_path (str): absolute path to image
        """
        data_uri = image_to_base64_data_uri(image_path)
        img = image(data_uri)

        chat = self.client.chat.create(model=self.model)

        chat.append(user(prompt, img))

        response = chat.sample()
        return response
