import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.core_plugins import ConversationSummaryPlugin
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.settings.open_ai_settings import OpenAISettings
from dotenv import load_dotenv

async def main():
    # Load environment variables from .env file (by default, it looks for .env in the current directory)
    load_dotenv()

    kernel = create_kernel()
    history = ChatHistory()

    chat_function = await create_chat_function(kernel)

    while True:
        try:
            request = input("User:> ")
        except KeyboardInterrupt:
            print("\n\nExiting chat...")
            return False
        except EOFError:
            print("\n\nExiting chat...")
            return False

        if request == "exit":
            print("\n\nExiting chat...")
            return False

        result = await kernel.invoke(
            chat_function,
            request=request,
            history=history,
        )

        # Add the request to the history
        history.add_user_message(request)
        history.add_assistant_message(str(result))

        print(f"Assistant:> {result}")

def create_kernel() -> sk.Kernel:
    # This automatically looks for OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, etc.
    settings = OpenAISettings()

    kernel = sk.Kernel()
    gpt = OpenAIChatCompletion(ai_model_id="gpt-4-turbo-preview", api_key=settings.api_key, org_id=settings.org_id, service_id="gpt4")
    kernel.add_service(gpt)

    # The following execution settings are used for the ConversationSummaryPlugin
    execution_settings = OpenAIChatPromptExecutionSettings(max_tokens=1024, temperature=0.1, top_p=0.5)
    
    prompt_template_config = PromptTemplateConfig(
        template="""
        text
        GIVEN A SECTION OF A CONVERSATION TRANSCRIPT, SUMMARIZE IT.

        CONTENT:
        {{$input}}

        SUMMARY:
        """,
        description="Given a section of a conversation transcript, summarize it",
        execution_settings={
            "gpt4": execution_settings,
            "default": execution_settings  # Optionally set a fallback
        }
    )

    # Import the ConversationSummaryPlugin
    kernel.add_plugin(ConversationSummaryPlugin(kernel=kernel, prompt_template_config=prompt_template_config),
                      plugin_name="ConversationSummaryPlugin")

    return kernel


async def create_chat_function(kernel: sk.Kernel) -> KernelFunction:   
    # Create the prompt with the ConversationSummaryPlugin
    prompt = """{{ConversationSummaryPlugin.SummarizeConversation $history}}
    User: {{$request}}
    Assistant:  """

    # These execution settings are tied to the chat function, created below.
    execution_settings = kernel.get_service("gpt4").instantiate_prompt_execution_settings(service_id="gpt4")
    chat_prompt_template_config = PromptTemplateConfig(
        template=prompt,
        description="Chat with the assistant",
        input_variables=[
            InputVariable(name="request", description="The user input", is_required=True),
            InputVariable(name="history", description="The history of the conversation", is_required=True),
        ],
    )

    # Create the function
    chat_function = kernel.add_function(
        prompt=prompt,
        plugin_name="Summarize_Conversation",
        function_name="Chat",
        description="Chat with the assistant",
        prompt_template_config=chat_prompt_template_config,
        execution_settings=execution_settings)
    
    return chat_function


if __name__ == "__main__":
    asyncio.run(main())
