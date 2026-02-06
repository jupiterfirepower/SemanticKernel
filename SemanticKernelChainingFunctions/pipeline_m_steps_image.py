import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.settings.open_ai_settings import OpenAISettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from OpenAiPlugins import Dalle3
from pydantic import ValidationError
from dotenv import load_dotenv

async def pipeline(kernel, function_list, input):
    for function in function_list:
        args = KernelArguments(input=input)
        input = await kernel.invoke(function, args)
    return input

async def main():
    try:
        # Load environment variables from .env file (by default, it looks for .env in the current directory)
        load_dotenv()

        kernel = sk.Kernel()

        # This automatically looks for OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, etc.
        settings = OpenAISettings()
        print(f"Chat Model: {settings.chat_model_id}")

        gpt35 = OpenAIChatCompletion("gpt-3.5-turbo", api_key=settings.api_key, org_id=settings.org_id, service_id="gpt35")
        kernel.add_service(gpt35)

        generate_image_plugin = kernel.add_plugin(Dalle3(), "Dalle3")
        animal_guesser = kernel.add_plugin(None, plugin_name="AnimalGuesser", parent_directory="./plugins")

        clues = """
            I am thinking of an animal.
            It is a mammal.
            It is a pet.
            It is a carnivore.
            It purrs."""

        function_list = [
            animal_guesser['GuessAnimal'],
            generate_image_plugin['ImageFromPrompt']
        ]

        animal_pic_url = await pipeline(kernel, function_list, clues)
        print(animal_pic_url)
    except ValidationError:
        print("Required OpenAI environment variables are missing.")

if __name__ == "__main__":
    asyncio.run(main())