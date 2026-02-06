import asyncio
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.settings.open_ai_settings import OpenAISettings
import semantic_kernel as sk
from dotenv import load_dotenv

async def main():
    kernel = sk.Kernel()
    # Load environment variables from .env file (by default, it looks for .env in the current directory)
    load_dotenv()

    # This automatically looks for OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, etc.
    settings = OpenAISettings()

    gpt35 = OpenAIChatCompletion("gpt-3.5-turbo", api_key=settings.api_key, org_id=settings.org_id, service_id = "gpt35")
    gpt4 = OpenAIChatCompletion("gpt-4", api_key=settings.api_key, org_id=settings.org_id, service_id = "gpt4")

    kernel.add_service(gpt35)
    kernel.add_service(gpt4)
    kernel.add_plugin(None, plugin_name="jokes", parent_directory="./plugins")

    prompt = "Create four knock-knock jokes: two about dogs, one about cats and one about ducks"

    # Configure the behavior
    # Set max tokens to 4000
    execution_settings = OpenAIChatPromptExecutionSettings(
        max_tokens=4000,
        #temperature=0.7,
        #top_p=0.8
    )

    # Set behavior to Auto
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Invoke the kernel with these settings
    result = await kernel.invoke_prompt(
        prompt=prompt,
        settings=execution_settings
    )

    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())