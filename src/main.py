from src.ai_controllers.ai_controller import AIController

ai_controller = AIController("gpt")
print(ai_controller.ask_model("Hello world!"))