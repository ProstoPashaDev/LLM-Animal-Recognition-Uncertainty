from src.ai_controllers.openai_controller import OpenAIController
from src.ai_controllers.grok_controller import XAIController
from src.metrics.NLL import NLL
from src.metrics.ECE import ECE
from xai_sdk.chat import image
from src.services.base64_image_encoder import image_to_base64_data_uri

#openai_controller = OpenAIController()
#print(ai_controller.ask_gpt("Hello world!"))
#xai_controller = XAIController()

image_path = "C://KhramovPavel/Project/Python/LLM-Animal-Recognition-Uncertainty/resources/animal.jpg"
data_uri = image_to_base64_data_uri(image_path)

'''
res = xai_controller.ask_grok_with_image("What is the animal on the picture? Provide a short answer. Express your "
                                         "confidence from 0 to 100",
                                         image(data_uri))
'''

"""
probs = [0.98, 0.9, 0.95, 0.83, 0.6712]
probs2 = [0.9, 0.3, 0.95, 0.83, 0.6712]
labels = [1, 0, 1, 1, 0]   # wrong answers get penalized heavily

nll = NLL()
ece = ECE()

print("ECE:", ece.compute_ece(probs, labels))
print("NLL:", nll.compute_nll(probs))
print("NLL2:", nll.compute_nll(probs2))
print(nll.get_avg_nll())
"""