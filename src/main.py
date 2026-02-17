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
from src.services.text_service import find_animal_confidense, split_animal_confidence, print_image_dataset

def experiment(prompt, animal_dataset, controller, unknown=False):
    print(controller.get_ai_name() + " experiment")
    print("-" * 20)

    accuracy = []
    confidence = []
    token_consumption = []
    answers = []

    for animal, image_path in animal_dataset.items():
        response = controller.ask_with_image(prompt, image_path)
        token_consumption.append(controller.get_tokens(response))
        answer = controller.get_text(response)
        answers.append(answer)
        #print("-"*20)
        #print(answer)
        #print("-"*20)
        answer = find_animal_confidense(answer)
        animal_ans, conf = split_animal_confidence(answer)

        if unknown:
            animal = "Unknown"

        accuracy.append(1) if animal_ans.lower() == animal.lower() else accuracy.append(0)
        print(answer + " | " + animal + " | " + str(accuracy[-1]))

        confidence.append(int(conf))

    print("Accuracy list:", accuracy)
    print("Confidence list:", confidence)
    print("Tokens consumption list:", token_consumption)
    print("Accuracy is", sum(accuracy) / len(accuracy))
    print("Mean confidence is", sum(confidence) / len(confidence))
    print("ECE:", compute_ece(confidence, accuracy))
    print("Tokens consumption:", sum(token_consumption))
    print("List of answers:", answers)
    print("-" * 20)


def experiment_grok(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object):
    xai_controller = XAIController()

    # experiment(prompt1, real_animals, xai_controller)
    # experiment(prompt2, real_animals, xai_controller)
    # experiment(prompt3, real_animals, xai_controller)

    # experiment(prompt1, hand_drawn_animals, xai_controller)
    # experiment(prompt2, hand_drawn_animals, xai_controller)
    # experiment(prompt3, hand_drawn_animals, xai_controller)

    # experiment(prompt1, hybrid_animals, xai_controller, unknown=True)
    # experiment(prompt2, hybrid_animals, xai_controller, unknown=True)
    # experiment(prompt3, hybrid_animals, xai_controller, unknown=True)

    # experiment(prompt1, random_object, xai_controller, unknown=True)
    # experiment(prompt2, random_object, xai_controller, unknown=True)
    # experiment(prompt3, random_object, xai_controller, unknown=True)


def experiment_claude(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object):
    claude_controller = ClaudeController()

    # experiment(prompt1, real_animals, claude_controller)
    # experiment(prompt2, real_animals, claude_controller)
    # experiment(prompt3, real_animals, claude_controller)

    # experiment(prompt1, hand_drawn_animals, claude_controller)
    # experiment(prompt2, hand_drawn_animals, claude_controller)
    # experiment(prompt3, hand_drawn_animals, claude_controller)

    # experiment(prompt1, hybrid_animals, claude_controller, unknown=True)
    # experiment(prompt2, hybrid_animals, claude_controller, unknown=True)
    # experiment(prompt3, hybrid_animals, claude_controller, unknown=True)

    # experiment(prompt1, random_object, claude_controller, unknown=True)
    # experiment(prompt2, random_object, claude_controller, unknown=True)
    # experiment(prompt3, random_object, claude_controller, unknown=True)

def experiment_chatgpt(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object):
    open_ai_controller = OpenAIController()

    # experiment(prompt1, real_animals, open_ai_controller)
    # experiment(prompt2, real_animals, open_ai_controller)
    # experiment(prompt3, real_animals, open_ai_controller)

    # experiment(prompt1, hand_drawn_animals, open_ai_controller)
    # experiment(prompt2, hand_drawn_animals, open_ai_controller)
    # experiment(prompt3, hand_drawn_animals, open_ai_controller)

    # experiment(prompt1, hybrid_animals, open_ai_controller, unknown=True)
    # experiment(prompt2, hybrid_animals, open_ai_controller, unknown=True)
    # experiment(prompt3, hybrid_animals, open_ai_controller, unknown=True)

    # experiment(prompt1, random_object, open_ai_controller, unknown=True)
    # experiment(prompt2, random_object, open_ai_controller, unknown=True)
    # experiment(prompt3, random_object, open_ai_controller, unknown=True)


def experiment_gemini(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object):
    gemini_controller = GeminiController()

    # experiment(prompt1, real_animals, gemini_controller)
    # experiment(prompt2, real_animals, gemini_controller)
    # experiment(prompt3, real_animals, gemini_controller)

    # experiment(prompt1, hand_drawn_animals, gemini_controller)
    # experiment(prompt2, hand_drawn_animals, gemini_controller)
    # experiment(prompt3, hand_drawn_animals, gemini_controller)

    # experiment(prompt1, hybrid_animals, gemini_controller, unknown=True)
    # experiment(prompt2, hybrid_animals, gemini_controller, unknown=True)
    # experiment(prompt3, hybrid_animals, gemini_controller, unknown=True)

    # experiment(prompt1, random_object, gemini_controller, unknown=True)
    # experiment(prompt2, random_object, gemini_controller, unknown=True)
    # experiment(prompt3, random_object, gemini_controller, unknown=True)



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
hand_drawn_animals = datasets.get_hand_drawn_AI_animals()
hybrid_animals = datasets.get_hybrid_animals()
random_object = datasets.get_random_objects(seed)

#print_image_dataset(real_animals)

# experiment_grok(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object)
# experiment_claude(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object)
# experiment_gemini(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object)
experiment_chatgpt(prompt1, prompt2, prompt3, real_animals, hand_drawn_animals, hybrid_animals, random_object)


