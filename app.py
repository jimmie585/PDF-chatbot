from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr
import os
import torch
import re
import time

# ==== Load Model and Embeddings ====
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever_global = None
recent_questions = []

# ==== PDF Processing ====
def process_pdf_with_cache(pdf_file):
    global retriever_global
    tmp_path = pdf_file.name
    file_id = os.path.splitext(os.path.basename(tmp_path))[0]
    cache_dir = f"vector_cache/{file_id}"
    faiss_file = os.path.join(cache_dir, "index.faiss")

    if os.path.exists(faiss_file):
        vectorstore = FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(cache_dir, exist_ok=True)
        vectorstore.save_local(cache_dir)

    retriever_global = vectorstore.as_retriever(search_kwargs={"k": 2})
    return f"✅ PDF '{file_id}' loaded. Ready to chat."

# ==== Format Bot Reply ====
def format_as_list_if_steps(text):
    lines = re.split(r'\n|(?<=\d\.)\s+', text)
    if len(lines) >= 2:
        return "\n".join(f"• {line.strip()}" for line in lines if line.strip())
    return text.strip()

# ==== Chat Handler ====
def ask_question_chat(message, chat_history):
    global retriever_global, recent_questions
    lower_msg = message.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    gratitude = ["thanks", "thank you"]
    identity = ["who are you", "what are you"]

    bot_reply = "🤖 Typing..."  # Placeholder
    chat_history.append((message, bot_reply))

    if message not in recent_questions:
        recent_questions.insert(0, message)
        recent_questions = recent_questions[:5]

    time.sleep(1.5)

    if lower_msg in greetings:
        bot_reply = "👋 Hello! I’m your PDF Assistant."
    elif lower_msg in gratitude:
        bot_reply = "You're welcome! 😊"
    elif lower_msg in identity:
        bot_reply = "🤖 I'm a smart assistant that can understand your documents and general queries."
    elif "summary" in lower_msg and retriever_global is not None:
        bot_reply = "📄 Here's your summary: [Download Summary](#)"
    elif retriever_global is None:
        prompt = f"Answer this as a helpful assistant:\n{message}"
        bot_reply = flan_pipeline(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
    else:
        relevant_docs = retriever_global.get_relevant_documents(message)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
You are a helpful assistant. Use the context to answer concisely.

Context:
{context}

Question: {message}

Answer:
""" if context.strip() else f"Answer this as a helpful assistant:\n{message}"
        bot_reply = flan_pipeline(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']

    bot_reply = format_as_list_if_steps(bot_reply)
    chat_history[-1] = (message, bot_reply)
    return "", chat_history, "\n".join(f"• {q}" for q in recent_questions)

# ==== Edit Message Handler ====
def edit_message(index, chat_history):
    if 0 <= index < len(chat_history):
        return chat_history[index][0]
    return ""

# ==== Clear Chat ====
def clear_chat():
    return [], ""

# ==== Instructions Toggle ====
def toggle_instructions(show):
    new_text = "❌ Hide Instructions" if not show else "❓ Show Instructions"
    new_visibility = not show
    return gr.update(visible=new_visibility), new_text, new_visibility

# ==== CSS for WhatsApp look ====
custom_css = """
.gradio-container {
    background-color: #ece5dd;
}
.message.bot {
    background-color: #ffffff !important;
    color: #000000 !important;
    padding: 10px;
    border-radius: 7.5px;
    margin: 4px;
    align-self: flex-start;
    max-width: 75%;
}
.message.user {
    background-color: #dcf8c6 !important;
    color: #000000 !important;
    padding: 10px;
    border-radius: 7.5px;
    margin: 4px;
    align-self: flex-end;
    max-width: 75%;
}
.chatbot {
    background-color: #e5ddd5;
    border-radius: 15px;
}
#upload-button, #submit-button {
    background-color: #25d366 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
}
.sidebar {
    background-color: #f7f7f7;
    padding: 15px;
    border-radius: 10px;
}
.sidebar .button {
    background-color: #25d366;
    color: white;
    font-weight: bold;
    border-radius: 6px;
    text-align: center;
    cursor: pointer;
    margin-bottom: 10px;
}
"""

# ==== Interface ====
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='color:#128C7E'>📄 WhatsApp-Style PDF Chatbot</h1>")

    instructions_box = gr.Markdown(
        value="""🔹 **Instructions:**
- Upload a PDF.
- Ask about its contents or anything general.
- Ask for summary, structure, or steps.
- Click "New Chat" to clear.
- 💬 You can click any message to edit & ask it again.""",
        visible=True
    )
    toggle_btn = gr.Button("❌ Hide Instructions")
    show_state = gr.State(True)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            new_chat_btn = gr.Button("🗑️ New Chat", elem_classes="button")
            gr.Markdown("### Recent")
            recent_display = gr.Markdown(value="")

        with gr.Column(scale=4):
            pdf_input = gr.File(label="📎 Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("📤 Process PDF", elem_id="upload-button")
            status_box = gr.Textbox(label="Status", interactive=False)

            chatbot = gr.Chatbot(label="💬 Chat", elem_classes="chatbot", height=500)
            question_input = gr.Textbox(
                show_label=False,
                placeholder="Type a message...",
                elem_id="question_input",
                lines=1
            )

    # ==== Function Bindings ====
    upload_btn.click(fn=process_pdf_with_cache, inputs=[pdf_input], outputs=[status_box])
    question_input.submit(fn=ask_question_chat, inputs=[question_input, chatbot], outputs=[question_input, chatbot, recent_display])
    new_chat_btn.click(fn=clear_chat, outputs=[chatbot, recent_display])
    toggle_btn.click(fn=toggle_instructions, inputs=[show_state], outputs=[instructions_box, toggle_btn, show_state])
    
    # Enable clicking on old messages to edit
    def handle_chat_click(evt: gr.SelectData, chat):
        index = evt.index[0]
        return chat[index][0]

    chatbot.select(fn=handle_chat_click, inputs=[chatbot], outputs=[question_input])

demo.launch(share=True)
gr.Interface(...).launch(server_name="0.0.0.0", server_port=10000)

