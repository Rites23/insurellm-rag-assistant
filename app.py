import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question


load_dotenv(override=True)



def format_context(docs):

    text = ""

    for doc in docs:

        text += (
            f"Source: {doc.metadata.get('source')}\n\n"
            f"{doc.page_content}\n"
            "------------------------\n\n"
        )

    return text



def chat(message, history):


    history = history or []


    answer, docs = answer_question(
        message,
        history
    )


    history.append(
        {
            "role": "user",
            "content": message
        }
    )


    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    return "", history, format_context(docs)




with gr.Blocks() as demo:


    gr.Markdown(
        "# 🏢 Insurellm Expert Assistant"
    )


    with gr.Row():


        chatbot = gr.Chatbot(
            height=500
        )


        context = gr.Textbox(
            label="Retrieved Context",
            lines=20
        )



    message = gr.Textbox(
        placeholder="Ask a question..."
    )



    message.submit(
        chat,
        inputs=[
            message,
            chatbot
        ],

        outputs=[
            message,
            chatbot,
            context
        ]
    )



demo.launch()