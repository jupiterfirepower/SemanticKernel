import asyncio
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.settings.open_ai_settings import OpenAISettings
from HomeAutomation import HomeAutomation
import semantic_kernel as sk
from dotenv import load_dotenv

async def fulfill_request(kernel: sk.Kernel, settings: OpenAIChatPromptExecutionSettings, request):
    print("Fulfilling request: " + request)

    # Invoke the kernel with these settings
    result = await kernel.invoke_prompt(
        prompt=request,
        settings=settings
    )

    print(result.final_answer)
    print("Request completed.\n\n")


async def main():
    # Load environment variables from .env file (by default, it looks for .env in the current directory)
    load_dotenv()

    # This automatically looks for OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, etc.
    settings = OpenAISettings()

    kernel = sk.Kernel()

    gpt4 = OpenAIChatCompletion("gpt-4", api_key=settings.api_key, org_id=settings.org_id, service_id = "gpt4")
    kernel.add_service(gpt4)

    kernel.add_plugin(HomeAutomation(), "HomeAutomation")
    kernel.add_plugin(None, plugin_name="MovieRecommender", parent_directory="./plugins")

    # Configure the behavior
    prompt_exec_settings = OpenAIChatPromptExecutionSettings(
        max_tokens=4000,
    )
    # Set behavior to Auto
    prompt_exec_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    await fulfill_request(kernel, prompt_exec_settings, "Turn on the lights in the kitchen")
    await fulfill_request(kernel, prompt_exec_settings, "Open the windows of the bedroom, turn the lights off and put on Shawshank Redemption on the TV.")
    await fulfill_request(kernel, prompt_exec_settings, "Close the garage door and turn off the lights in all rooms.")
    await fulfill_request(kernel, prompt_exec_settings, "Turn off the lights in all rooms and play a movie in which Tom Cruise is a lawyer in the living room.")

if __name__ == "__main__":
    asyncio.run(main())