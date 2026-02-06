import asyncio
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
import semantic_kernel as sk
from CheckSpreadsheet import CheckSpreadsheet
from ParseWordDocument import ParseWordDocument
from semantic_kernel.functions.kernel_arguments import KernelArguments

async def run_spreadsheet_check(path, function):
    kernel = sk.Kernel()
    print("Document - ", path)
    check_spreadsheet = kernel.add_plugin(CheckSpreadsheet(), "CheckSpreadsheet")

    result = await kernel.invoke(
        check_spreadsheet[function], KernelArguments(path = path)
    )
    print(result)


async def run_document_check(path, function, target_heading, semantic_function):
    kernel = sk.Kernel()
    ollama_chat_service = OllamaChatCompletion(
        service_id="ollama_chat",
        ai_model_id="llama3.2:3b",
        host="http://localhost:11434"
    )

    kernel.add_service(ollama_chat_service)

    parse_word_document = kernel.add_plugin(ParseWordDocument(), "ParseWordDocument")

    text = await kernel.invoke(
        parse_word_document[function],
        KernelArguments(doc_path = path, target_heading = target_heading)
    )

    check_docs = kernel.add_plugin(None, "ProposalChecker", "./plugins")
    result = await kernel.invoke(check_docs[semantic_function], KernelArguments(input = str(text)))
    print(f"{target_heading}: {result}")

async def main():
    data_path = "./data/proposals"
    await run_spreadsheet_check(f"{data_path}/correct/correct.xlsx", "check_values")
    await run_spreadsheet_check(f"{data_path}/incorrect01/incorrect_template.xlsx", "check_tabs")
    await run_spreadsheet_check(f"{data_path}/incorrect02/over_budget.xlsx", "check_values")
    await run_spreadsheet_check(f"{data_path}/incorrect03/fast_increase.xlsx", "check_values")
    
    # await run_spreadsheet_check(f"{data_path}/correct/correct.xlsx", "CheckCells")
    # await run_spreadsheet_check(f"{data_path}/incorrect04/incorrect_cells.xlsx", "CheckCells")
    # await run_spreadsheet_check(f"{data_path}/incorrect02/over_budget.xlsx", "CheckCells")
    # await run_spreadsheet_check(f"{data_path}/incorrect03/fast_increase.xlsx", "CheckCells")

    # await run_spreadsheet_check(f"{data_path}/correct/correct.xlsx", "CheckValues")
    # await run_spreadsheet_check(f"{data_path}/incorrect02/over_budget.xlsx", "CheckValues")
    # await run_spreadsheet_check(f"{data_path}/incorrect03/fast_increase.xlsx", "CheckValues")

    word_doc_path = f"{data_path}/correct/correct.docx"
    fun_name = "extract_text_under_heading"

    print("Document - ", word_doc_path)
    print("Word document checks:")

    await run_document_check(word_doc_path, fun_name, "Experience", "CheckExperience")
    await run_document_check(word_doc_path, fun_name, "Team", "CheckQualifications")
    await run_document_check(word_doc_path, fun_name, "Implementation", "CheckImplementationDescription")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())