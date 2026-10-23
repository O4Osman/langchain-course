import os
from operator import itemgetter

from dotenv import load_dotenv

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from pinecone import Pinecone
from pinecone.models.documents.score_by import EmbedQuery
from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3.2:3b', temperature=0)

print("Initializing Pinecone...")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.index(name=os.environ["PINECONE_INDEX"])

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    """
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.chunk_text for doc in docs)


def retriever(query: str, k=3):
    """Retrieves similar chunks"""

    results = index.documents.search(
        namespace="langchain-course",
        top_k=k,
        score_by=[
            EmbedQuery(field="chunk_text", query=query)
        ],
        include_fields=["chunk_text", "title", "source"],
    )

    return results.matches


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever(query)

    # Step 2: Format document into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLm with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
            RunnablePassthrough.assign(
                context=(
                        RunnableLambda(itemgetter("question"))
                        | retriever
                        | RunnableLambda(format_docs)
                )
            )
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is pinecone in machine learning?"

    # # ========================================================================
    # # Option 0: Raw invocation without RAG
    # # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)
    #
    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 1: Without LCEL")
    # print("=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer:")
    # print(result_without_lcel)
    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_wiht_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_wiht_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
