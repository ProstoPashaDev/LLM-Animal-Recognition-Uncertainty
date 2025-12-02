import os

import anthropic

from src.services.base64_image_encoder import image_to_base64_data_uri


class GeminiController:

    def __init__(self):
        self.model = "claude-opus-4-5"
        key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)

    def ask_claude_with_image(self, prompt: str, image_path: str):
        image_data = image_to_base64_data_uri(image_path)

        result = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        return result

