from langchain_community.llms.ollama import Ollama
import click
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

@click.command()
@click.option("-s", "--system-prompt", "systemPrompt", required=True, type=str, help="System prompt to provide to the AI model",)
@click.option("-p", "--prompt", "prompt", required=True, type=str, help="Prompt to provide to the AI model",)
def main(systemPrompt: str, prompt: str)    ->  None:
    output_parser = StrOutputParser()
    chatPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", systemPrompt),
        ("user", "{input}")
    ])

    llm: Ollama = Ollama(model="llama2")
    
    chain: RunnableSequence = chatPrompt | llm | output_parser

    print(chain.invoke({"input": prompt}))


if __name__ == "__main__":
    main()
