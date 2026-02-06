import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding, OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments, KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai.settings.open_ai_settings import OpenAISettings
from dotenv import load_dotenv
from semantic_kernel.connectors.in_memory import InMemoryStore

from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel, DistanceFunction
from typing import Annotated
from dataclasses import dataclass

@vectorstoremodel
@dataclass
class MyRecord:
    text: Annotated[str, VectorStoreField('data', is_indexed=True, is_full_text_indexed=True)]
    id: Annotated[str, VectorStoreField('key')]
    vector: Annotated[list[float] | str | None, VectorStoreField(
        'vector',
        dimensions=1536,
        distance_function= DistanceFunction("cosine"),
        embedding_generator=OpenAITextEmbedding(ai_model_id="text-embedding-3-small"),
    )] = None

    def __post_init__(self):
        if self.vector is None:
            self.vector = f"{self.text}"


async def add_to_memory(memory_store: InMemoryStore, id_str: str, text: str):
    collection = memory_store.get_collection(collection_name="generic", record_type=MyRecord)
    records = [
        MyRecord(id=id_str, text=text),
    ]
    # this method adds vectors automatically
    await collection.upsert(records)

def create_kernel() -> tuple[sk.Kernel, OpenAITextEmbedding]:
    # Load environment variables from .env file (by default, it looks for .env in the current directory)
    load_dotenv()

    # This automatically looks for OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, etc.
    settings = OpenAISettings()

    kernel = sk.Kernel()
    gpt = OpenAIChatCompletion(ai_model_id="gpt-4-turbo-preview", api_key=settings.api_key, org_id=settings.org_id, service_id="gpt4")
    emb = OpenAITextEmbedding(ai_model_id="text-embedding-ada-002", api_key=settings.api_key, org_id=settings.org_id, service_id="emb")
    kernel.add_service(emb)
    kernel.add_service(gpt)
    return kernel, emb

async def tour(kernel: sk.Kernel) -> KernelFunction:
    prompt = """
    Information about me, from previous conversations:
    - {{$city}} {{recall $city}}
    - {{$activity}} {{recall $activity}}
    """.strip()

    execution_settings = kernel.get_service("gpt4").instantiate_prompt_execution_settings(service_id="gpt4")
    execution_settings.max_tokens = 4000
    execution_settings.temperature = 0.8

    prompt_template_config = PromptTemplateConfig(template=prompt)

    chat_func = kernel.add_function(
        function_name="chat_with_memory",
        #plugin_name="TextMemoryPlugin",
        plugin_name="memory",
        prompt_template_config=prompt_template_config,
        execution_settings=execution_settings
    )

    return chat_func


async def main():
    kernel, emb = create_kernel()

    memory_store = InMemoryStore()
    collection = memory_store.get_collection(collection_name="generic", record_type=MyRecord)
    await collection.collection_exists(collection_name="generic", record_type=MyRecord)

    # Create search function directly on collection
    search_function = collection.create_search_function(
        function_name="search",
        search_type="vector",  # or "keyword_hybrid"
        top=10,
        vector_property_name="vector",  # Name of the vector field
    )

    # Add to kernel directly
    kernel.add_function(plugin_name="memory", function=search_function)

    await add_to_memory(memory_store, id_str="1", text="My favorite city is Paris")
    await add_to_memory(memory_store, id_str="2", text="My favorite activity is visiting museums")

    f = await tour(kernel)

    args = KernelArguments()
    args["city"] = "My favorite city is Paris"
    args["activity"] = "My favorite activity is visiting museums" 
    answer = await kernel.invoke(f, arguments=args)
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
