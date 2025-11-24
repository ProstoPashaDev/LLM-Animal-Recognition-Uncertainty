from src.ai_controllers.openai_controller import OpenAIController
from src.ai_controllers.grok_controller import XAIController
from src.metrics.NLL import NLL
from src.metrics.ECE import ECE

#openai_controller = OpenAIController()
#print(ai_controller.ask_gpt("Hello world!"))
#xai_controller = XAIController()
#print(xai_controller.ask_grok("What is the meaning of life, the universe, and everything?"))

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