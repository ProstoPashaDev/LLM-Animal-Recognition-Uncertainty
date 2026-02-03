from random import Random

from src.ai_controllers.openai_controller import OpenAIController
from src.ai_controllers.xai_controller import XAIController
from src.ai_controllers.gemini_controller import GeminiController
from src.ai_controllers.claude_controller import ClaudeController
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


def experiment(prompt, animal_dataset, controller, unknown=False):
    #print("Grok experiment")
    print("Claude experiment")
    print("-" * 20)

    accuracy = []
    confidence = []
    token_consumption = []
    answers = []

    for animal, image_path in animal_dataset.items():
        response = controller.ask_with_image(prompt, image_path)
        #token_consumption.append(response.usage.total_tokens) #For grok
        token_consumption.append(response.usage.input_tokens + response.usage.output_tokens)  #For Claude
        #answer = response.content #For Grok
        answer = response.content[0].text  #For Claude
        answers.append(answer)
        #print("-"*20)
        #print(answer)
        #print("-"*20)
        if answer.count("\n") > 1:
            ans1 = answer.split("\n")[0]
            ans2 = answer.split("\n")[-1]
            #print(ans1, ans2)
            if len(ans1.split(" ")) == 2:
                answer = ans1
            elif len(ans2.split(" ")) == 2:
                answer = ans2
            else:
                dig = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}
                l = 0
                r = 0
                for i in range(len(answer)):
                    if answer[i] in dig:
                        l = i - 2
                        r = i
                        while r < len(answer) and (answer[r] != " " and answer[r] != "." and answer[r] != "\n" and answer[r] != "*"):
                            r += 1

                        while l > 0 and (answer[l] != " " and answer[l] != "." and answer[l] != "\n" and answer[l] != "*"):
                            l -= 1

                        break

                answer = answer[l+1:r]

        #print(answer)
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
            conf = conf[:len(conf) - 1:]
        try:
            confidence.append(int(conf))
        except:
            pass

    print("Accuracy list:", accuracy)
    print("Confidence list:", confidence)
    print("Tokens consumption list:", token_consumption)
    print("Accuracy is", sum(accuracy) / len(accuracy))
    print("Mean confidence is", sum(confidence) / len(confidence))
    print("ECE:", compute_ece(confidence, accuracy))
    print("Tokens consumption:", sum(token_consumption))
    print("List of answers:", answers)
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

real_animals = datasets.get_real_animals(seed)
#for key, value in real_animals.items():
#print(key, value.split("\\")[-1])

#experiment_grok(prompt1, real_animals)
#experiment_grok(prompt2, real_animals)
#experiment_grok(prompt3, real_animals)

claude_controller = ClaudeController()

#experiment(prompt1, real_animals, claude_controller)
#experiment(prompt2, real_animals, claude_controller)
#experiment(prompt3, real_animals, claude_controller)

hand_drawn_animals = datasets.get_hand_drawn_AI_animals()

#experiment_grok(prompt1, hand_drawn_animals)
#experiment_grok(prompt2, hand_drawn_animals)
#experiment_grok(prompt3, hand_drawn_animals)

#experiment(prompt1, hand_drawn_animals, claude_controller)
#experiment(prompt2, hand_drawn_animals, claude_controller)
#experiment(prompt3, hand_drawn_animals, claude_controller)

hybrid_animals = datasets.get_hybrid_animals()

#experiment_grok(prompt1, hybrid_animals, unknown=True)
#experiment_grok(prompt2, hybrid_animals, unknown=True)
#experiment_grok(prompt3, hybrid_animals, unknown=True)

#experiment(prompt1, hybrid_animals, claude_controller, unknown=True)
#experiment(prompt2, hybrid_animals, claude_controller, unknown=True)
experiment(prompt3, hybrid_animals, claude_controller, unknown=True)

random_object = datasets.get_random_objects(seed)

#experiment_grok(prompt1, random_object, unknown=True)
#experiment_grok(prompt2, random_object, unknown=True)
#experiment_grok(prompt3, random_object, unknown=True)

#experiment(prompt1, random_object, claude_controller, unknown=True)
#experiment(prompt2, random_object, claude_controller, unknown=True)
#experiment(prompt3, random_object, claude_controller, unknown=True)
