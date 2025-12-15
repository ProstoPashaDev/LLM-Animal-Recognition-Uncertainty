from random import Random

from src.ai_controllers.openai_controller import OpenAIController
from src.ai_controllers.xai_controller import XAIController
from src.image_repositories.datasets_controller import DatasetsController
from src.metrics.NLL import NLL
from src.metrics.ECE import compute_ece
from xai_sdk.chat import image
from src.services.base64_image_encoder import image_to_base64_data_uri

"""
#openai_controller = OpenAIController()
#print(ai_controller.ask_gpt("Hello world!"))
#xai_controller = XAIController()

image_path = "C://KhramovPavel/Project/Python/LLM-Animal-Recognition-Uncertainty/resources/animal.jpg"

'''
res = xai_controller.ask_grok_with_image("What is the animal on the picture? Provide a short answer. Express your "
                                         "confidence from 0 to 100",
                                         image_path)
'''

"""
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


def experiment_grok(prompt, animal_dataset, unknown=False):
    print("Grok experiment")
    print("-" * 20)
    xai_controller = XAIController()

    accuracy = []
    confidence = []
    token_consumption = []

    for animal, image_path in animal_dataset.items():
        response = xai_controller.ask_grok_with_image(prompt, image_path)
        token_consumption.append(response.usage.total_tokens)
        answer = response.content
        try:
            animal_ans = answer.split(",")[0]
            conf = answer.split(",")[1]
        except:
            animal_ans = answer.split(" ")[0]
            conf = answer.split(" ")[1]

        if unknown:
            animal = "Unknown"

        accuracy.append(1) if animal_ans.lower() == animal.lower() else accuracy.append(0)
        print(answer + " | " + animal + " | " + str(accuracy[-1]))
        if conf[-1] == "%":
            conf = conf[:len(conf)-1:]
        confidence.append(int(conf))

    print("Accuracy list:", accuracy)
    print("Confidence list:", confidence)
    print("Tokens consumption list:", token_consumption)
    print("Grok accuracy is", sum(accuracy) / len(accuracy))
    print("Grok mean confidence is", sum(confidence) / len(confidence))
    print("ECE:", compute_ece(confidence, accuracy))
    print("Tokens consumption:", sum(token_consumption))
    print("-" * 20)


datasets = DatasetsController()
prompt1 = ("Which real animal is in the image? Provide a general name of the animal, not a concrete species. Indicate "
           "your certainty from 0 to 100. Output your response in the following format: animal, certainty. If you are "
           "unable to identify the real animal in the image enter: unknown, <certainty>.")

prompt2 = ("Identify the real animal depicted in the provided image. Provide a general name of the animal, "
           "not a concrete species. Indicate your certainty from 0 to 100. Output your response in the following "
           "format: animal, certainty. If you are unable to identify the real animal in the image enter: unknown, "
           "certainty.")

prompt3 = ("Which real animal is in the image? Provide a general name of the animal, not a concrete species. Indicate "
           "your certainty from 0 to 100. Output your response in the following format: animal, certainty. If you are "
           "unable to identify the real animal in the image enter: unknown, certainty. Response example: lion, 95.")

seed = 1208219

#real_animals = datasets.get_real_animals(seed)
#for key, value in real_animals.items():
    #print(key, value.split("\\")[-1])

#experiment_grok(prompt1, real_animals)
#experiment_grok(prompt2, real_animals)
#experiment_grok(prompt3, real_animals)

hand_drawn_animals = datasets.get_hand_drawn_AI_animals()

#experiment_grok(prompt1, hand_drawn_animals)
#experiment_grok(prompt2, hand_drawn_animals)
#experiment_grok(prompt3, hand_drawn_animals)

hybrid_animals = datasets.get_hybrid_animals()

#experiment_grok(prompt1, hybrid_animals, unknown=True)
#experiment_grok(prompt2, hybrid_animals, unknown=True)
#experiment_grok(prompt3, hybrid_animals, unknown=True)

random_object = datasets.get_random_objects(seed)

#experiment_grok(prompt1, random_object, unknown=True)
#experiment_grok(prompt2, random_object, unknown=True)
#experiment_grok(prompt3, random_object, unknown=True)
