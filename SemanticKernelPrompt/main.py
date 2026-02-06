import semantic_kernel as sk
# from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
import asyncio

from semantic_kernel.functions import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import  kernel_function

import random
class ShowManager:
  @kernel_function(description="Randomly choose among a theme for a joke",name="random_theme")
  def random_theme(self) -> str:
     themes = ["Boo", "Dishes", "Art", "Needle", "Tank", "Police"]
     theme = random.choice(themes)
     return theme

async def main():
    kernel = sk.Kernel()

    #gpt35 = OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
    #gpt4 = OpenAIChatCompletion("gpt-4", api_key, org_id)
    #kernel.add_service(gpt35)
    #kernel.add_service(gpt4)

    kernel.add_service(
        OllamaChatCompletion(
            service_id="ollama_chat",
            ai_model_id="llama3.2:3b",
            host="http://localhost:11434"
        )
    )

    prompt = "Finish the following knock-knock joke. Knock, knock. Who's there? Dishes. Dishes who?"

    prompt_function = kernel.add_function(function_name="ex01",
                                          plugin_name="sample", prompt=prompt)

    #response = await kernel.invoke(prompt_function, request=prompt)
    #print(response)

    prompt = "Finish the following knock-knock joke. Knock, knock. Who 's there? {{$input}}, {{$input}} who?"


    #args = KernelArguments(input="Boo")
    #response = await kernel.invoke(prompt_function, request=prompt, arguments=args)
    #print(response)

    theme_choice = kernel.add_plugin(ShowManager(), "ShowManager")

    response = await kernel.invoke(theme_choice["random_theme"])
    print(response)

    # 3. Define a prompt
    #prompt = "Why is the sky blue? Explain it in simple terms."

    # 4. Get the chat completion service
    #chat_service = kernel.get_service("ollama_chat")

    # 5. Invoke the model
    #print(f"Request: {prompt}")
    #response = await chat_service.get_chat_message_content_async(prompt)
    #print(f"Response: {response}")

# from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
# from semantic_kernel.functions.kernel_arguments import KernelArguments

#from semantic_kernel.functions import kernel_function

#class PromptEngineeringPlugin:
#    @kernel_function(name="attractions_single_variable")
#    def attractions_single_variable(self, input: str) -> str:
#        return f"Processed: {input}"

async def main_plug(name):
    kernel_plug = sk.Kernel()

    # Registering
    #kernel_plug.add_plugin(PromptEngineeringPlugin(), plugin_name="prompt_engineering")

    kernel_plug.add_service(
        OllamaChatCompletion(
            service_id="ollama",
            ai_model_id="llama3.2:3b",
            host="http://localhost:11434"
        )
    )

    pe_plugin = kernel_plug.add_plugin(None, parent_directory="./plugins", plugin_name="prompt_engineering")
    response_plug = await kernel_plug.invoke(pe_plugin[name], KernelArguments(city="New York City"))
    print(response_plug)
    print("-" * 40)

async def main_with_parms():
    kernel_plug_params = sk.Kernel()

    kernel_plug_params.add_service(
        OllamaChatCompletion(
            service_id="ollama",
            ai_model_id="llama3.2:3b",
            host="http://localhost:11434"
        )
    )

    pe_plugin = kernel_plug_params.add_plugin(None, parent_directory="./plugins", plugin_name="prompt_engineering")
    response_params = await kernel_plug_params.invoke(pe_plugin["attractions_multiple_variables"], KernelArguments(city = "New York City",
       n_days = "3",
       likes = "restaurants, Ghostbusters, Friends tv show",
       dislikes = "museums, parks",
       n_attractions = "5"
    ))
    print(response_params)
    print("-" * 40)

async def main_with_cot():
    kernel_plug_cot = sk.Kernel()

    kernel_plug_cot.add_service(
        OllamaChatCompletion(
            service_id="ollama",
            ai_model_id="llama3.2:3b",
            host="http://localhost:11434"
        )
    )

    pe_plugin = kernel_plug_cot.add_plugin(None, parent_directory="./plugins", plugin_name="prompt_engineering")

    problem = """When I was 6 my sister was half my age. Now I'm 70. 
    How old is my sister?"""
    solve_steps = await kernel_plug_cot.invoke(pe_plugin["solve_math_problem_v2"], KernelArguments(problem = problem))
    print(f"\n\nSteps: {str(solve_steps)}\n\n")
    print("*" * 40)

    response = await kernel_plug_cot.invoke(pe_plugin["chain_of_thought"], KernelArguments(problem=problem, input=str(solve_steps)))
    print(f"\n\nFinal answer: {str(response)}\n\n")

import re


def extract_numbers_regex(text):
    # Pattern to match integers, floats, and potentially negative numbers
    # r'-?\d*\.?\d+' matches:
    # -? : an optional negative sign
    # \d* : zero or more digits
    # \.? : an optional decimal point
    # \d+ : one or more digits
    matches = re.findall(r'-?\d*\.?\d+', text)

    # Convert the extracted string numbers to float or int as appropriate
    numbers = []
    for x in matches:
        if '.' in x:
            numbers.append(float(x))
        else:
            numbers.append(int(x))
    return numbers


async def main_with_cot2_deepseek():
    kernel_plug_cot2 = sk.Kernel()

    kernel_plug_cot2.add_service(
        OllamaChatCompletion(
            service_id="ollama",
            ai_model_id="deepseek-r1:1.5b",
            host="http://localhost:11434"
        )
    )

    problem = """When I was 6 my sister was half my age. Now I'm 70. 
                 How old is my sister?"""

    pe_plugin = kernel_plug_cot2.add_plugin(None, parent_directory="./plugins", plugin_name="prompt_engineering")

    responses = []
    for i in range(7):
        solve_steps = await kernel_plug_cot2.invoke(pe_plugin["solve_math_problem_v2"], KernelArguments(problem = problem))
        response = await kernel_plug_cot2.invoke(pe_plugin["chain_of_thought_v2"], KernelArguments(problem = problem, input = str(solve_steps)))
        print("$" * 40)
        numbers_list = extract_numbers_regex(str(response))
        print(numbers_list[-1])
        print("!" * 40)
        print(response)
        responses.append(int(str(numbers_list[-1])))
        #responses.append(int(str(response)))

    print("Responses:")
    print(responses)
    final_answer = max(set(responses), key=responses.count)
    print(f"Final answer: {final_answer}")

async def main_with_cot2_llama3_2_reasoning():
    kernel_plug_cot2 = sk.Kernel()

    kernel_plug_cot2.add_service(
        OllamaChatCompletion(
            service_id="ollama",
            #ai_model_id="llama3.2:3b",
            ai_model_id="hf.co/mradermacher/DeepThought-MOE-8X3B-R1-Llama-3.2-Reasoning-18B-GGUF",
            host="http://localhost:11434"
        )
    )

    problem = """When I was 6 my sister was half my age. Now I'm 70. 
                 How old is my sister?"""

    pe_plugin = kernel_plug_cot2.add_plugin(None, parent_directory="./plugins", plugin_name="prompt_engineering")

    responses = []
    for i in range(7):
        solve_steps = await kernel_plug_cot2.invoke(pe_plugin["solve_math_problem_v2"], KernelArguments(problem = problem))
        response = await kernel_plug_cot2.invoke(pe_plugin["chain_of_thought_v2"], KernelArguments(problem = problem, input = str(solve_steps)))
        #print("$" * 40)
        #numbers_list = extract_numbers_regex(str(response))
        #print(numbers_list[-1])
        #print("!" * 40)
        #print(response)
        #responses.append(int(str(numbers_list[-1])))
        responses.append(int(str(response)))

    print("Responses:")
    print(responses)
    final_answer = max(set(responses), key=responses.count)
    print(f"Final answer: {final_answer}")

async def main_time_plugin():
    kernel_plug_time = sk.Kernel()

    kernel_plug_time.add_service(
        OllamaChatCompletion(
            service_id="ollama",
            ai_model_id="llama3.2:3b",
            host="http://localhost:11434"
        )
    )

    kernel_plug_time.add_plugin(TimePlugin(), "time")

    prompt = """
        Today is: {{time.date}}
        Current time is: {{time.time}}
        Answer to the following questions using JSON syntax, including the
    data used.
        Is it morning, afternoon, evening, or night (morning/afternoon/
    evening/night)?
        Is it weekend time (weekend/not weekend)?
        """
    prompt_function = kernel_plug_time.add_function(function_name="ex03", plugin_name="sample", prompt=prompt)
    response = await kernel_plug_time.invoke(prompt_function, request=prompt)

    print(response)

from semantic_kernel.core_plugins.time_plugin import TimePlugin

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #asyncio.run(main())
    #asyncio.run(main_plug("attractions_single_variable"))
    #asyncio.run(main_plug("attractions_single_variable_v2"))
    #asyncio.run(main_with_parms())
    #asyncio.run(main_with_cot())
    #asyncio.run(main_with_cot2_deepseek())
    #asyncio.run(main_with_cot2_llama3_2_reasoning())
    asyncio.run(main_time_plugin())


