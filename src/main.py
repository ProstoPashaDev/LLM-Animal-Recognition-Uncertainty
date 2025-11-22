from src.ai_controllers.openai_controller import OpenAIController
from src.ai_controllers.grok_controller import XAIController

openai_controller = OpenAIController()
#print(ai_controller.ask_gpt("Hello world!"))
xai_controller = XAIController()
#print(xai_controller.ask_grok("What is the meaning of life, the universe, and everything?"))

