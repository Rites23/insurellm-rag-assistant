import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


load_dotenv(override=True)



MODEL = "gpt-4.1-nano"


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)



vectorstore = PineconeVectorStore(
    index_name="insurellm-main",
    embedding=embeddings,
    namespace="insurellm"
)



retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5
    }
)



llm = ChatOpenAI(
    temperature=0,
    model=MODEL
)



SYSTEM_PROMPT = """

You are an AI assistant for Insurellm.

Answer questions using only the provided context.

If the answer is not available, say you do not know.

Context:

{context}

"""



def answer_question(question, history=None):


    docs = retriever.invoke(question)


    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    messages = [

        SystemMessage(
            content=SYSTEM_PROMPT.format(
                context=context
            )
        )

    ]


    if history:

        for message in history:

            if message["role"] == "user":

                messages.append(
                    HumanMessage(
                        content=message["content"]
                    )
                )


            elif message["role"] == "assistant":

                messages.append(
                    AIMessage(
                        content=message["content"]
                    )
                )



    messages.append(
        HumanMessage(
            content=question
        )
    )


    response = llm.invoke(messages)



    return response.content, docs