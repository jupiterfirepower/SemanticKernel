import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.settings.open_ai_settings import OpenAISettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from OpenAiPlugins import Dalle3
from dotenv import load_dotenv

async def main():
    # Load environment variables from .env file (by default, it looks for .env in the current directory)
    load_dotenv()

    kernel = sk.Kernel()

    # This automatically looks for OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, etc.
    settings = OpenAISettings()

    gpt35 = OpenAIChatCompletion(ai_model_id="gpt-3.5-turbo", api_key=settings.api_key, org_id=settings.org_id, service_id="gpt35")
    kernel.add_service(gpt35)

    dalle3 = kernel.add_plugin(Dalle3(), "Dalle3")
    animal_str = "A painting of a cat sitting on a sofa in the impressionist style"

    animal_pic_url = await kernel.invoke(dalle3['ImageFromPrompt'], KernelArguments(input = animal_str))
    print(animal_pic_url)

if __name__ == "__main__":
    asyncio.run(main())