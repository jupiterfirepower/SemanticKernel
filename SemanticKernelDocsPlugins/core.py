import asyncio
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
import semantic_kernel as sk
from semantic_kernel.core_plugins.time_plugin import TimePlugin

async def main():

    kernel = sk.Kernel()
    ollama_chat_service = OllamaChatCompletion(
            service_id="ollama_chat",
            ai_model_id="llama3.2:3b",
            host="http://localhost:11434"
        )

    kernel.add_service(ollama_chat_service)
    kernel.add_plugin(TimePlugin(), "time")

    prompt  = """
    Today is: {{time.date}}
    Current time is: {{time.time}}

    Answer to the following questions using JSON syntax, including the data used.
    Is it morning, afternoon, evening, or night (morning/afternoon/evening/night)?
    Is it weekend time (weekend/not weekend)?
    """
    prompt_function = kernel.add_function(function_name="ex01", plugin_name="sample", prompt=prompt)
    response = await kernel.invoke(prompt_function, request=prompt)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())