"""
Gradio UI for Mistral 7B with RAG
"""

import os
from typing import List

import gradio as gr
from langchain_core.runnables.base import RunnableSequence
import numpy as np
from confluence_rag import generate_rag_chain, load_pdf, store_vector, load_multiple_pdf


def initialize_chain(file: gr.File) -> RunnableSequence:
    """
    Initializes the chain with the given file.
    If no file is provided, the llm is used without RAG.
    Args:
        file (gr.File): file to initialize the chain with
    Returns:
        RunnableSequence: the chain
    """
    if file is None:
        return generate_rag_chain()

    if len(file) == 1:
        pdf = load_pdf(file[0].name)
    else:
        pdf = load_multiple_pdf([f.name for f in file])
    retriever = store_vector(pdf)

    return generate_rag_chain(retriever)


def invoke_chain(message: str, history: List[str], file: gr.File = None) -> str:
    """
    Invokes the chain with the given message and updates the chain if a new file is provided.
    Args:
        message (str): message to invoke the chain with
        history (List[str]): history of messages
        file (gr.File, optional): file to update the chain with. Defaults to None.
    Returns:
        str: the response of the chain
    """
    # Check if file is provided and exists
    if file is not None and not np.all([os.path.exists(f.name) for f in file]) or len(file) == 0:
        return "Error: File not found."

    if file is not None and not np.all([f.name.endswith(".pdf") for f in file]):
        return "Error: File is not a pdf."

    chain = initialize_chain(file)
    return chain.invoke(message)


def create_demo() -> gr.Interface:
    """
    Creates and returns a Gradio Chat Interface.
    Returns:
        gr.Interface: the Gradio Chat Interface
    """
    return gr.ChatInterface(
        invoke_chain,
        additional_inputs=[gr.File(label="File", file_count='multiple')],
        title="Mistral 7B with RAG",
        description="Ask questions to Mistral about your pdf document.",
        theme="soft",
    )


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()