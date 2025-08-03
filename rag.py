from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr
import tempfile
import os
import torch

# ==== 🔧 Load Model + Embeddings ====
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
flan_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever_global = None  # Global retriever cache


# ==== 📄 PDF Processor ====
def clean_filename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")


def process_pdf_with_cache(pdf_file):
    global retriever_global
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    file_id = clean_filename(pdf_file.name)
    cache_dir = f"vector_cache/{file_id}"
    faiss_file = os.path.join(cache_dir, "index.faiss")

    if os.path.exists(faiss_file):
        vectorstore = FAISS.load_local(cache_dir, embeddings)
    else:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(cache_dir, exist_ok=True)
        vectorstore.save_local(cache_dir)

    retriever_global = vectorstore.as_retriever(search_kwargs={"k": 2})
    return f"✅ PDF '{file_id}' loaded and ready to chat."


# ==== 💬 Question Answering ====
def ask_question_chat(message, chat_history):
    if retriever_global is None:
        return "⚠️ Please upload and process a PDF first.", chat_history

    relevant_docs = retriever_global.get_relevant_documents(message)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a helpful assistant. Answer the question using only the context below.

Context:
{context}

Question: {message}

Answer:
"""

    response = flan_pipeline(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
    chat_history.append((message, response))
    return "", chat_history


# ==== 🎨 WhatsApp-style CSS ====
custom_css = """
.gradio-container { background-color: #e5ddd5; font-family: 'Segoe UI', sans-serif; }

.message.bot {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 10px 10px 10px 0;
    padding: 10px;
    margin: 4px 0;
    max-width: 75%;
    text-align: left;
}

.message.user {
    background-color: #dcf8c6 !important;
    color: #000000 !important;
    border-radius: 10px 10px 0 10px;
    padding: 10px;
    margin: 4px 0;
    max-width: 75%;
    text-align: right;
    align-self: flex-end;
}

#upload-button, #submit-button {
    background-color: #128C7E !important;
    color: white !important;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}

.chatbot {
    background-color: #e5ddd5;
    padding: 10px;
    border-radius: 12px;
}
"""


# ==== 🧠 Gradio UI ====
with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("<h1 style='color:#075E54'>📱 Chat with Your PDF — WhatsApp Style</h1>")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("📄 Process PDF", elem_id="upload-button")

    status_box = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="PDF Assistant", height=500, bubble_full_width=False)

    with gr.Row():
        question_input = gr.Textbox(show_label=False, placeholder="Ask your question...")
        submit_btn = gr.Button("Ask", elem_id="submit-button")

    upload_btn.click(fn=process_pdf_with_cache, inputs=[pdf_input], outputs=[status_box])
    submit_btn.click(fn=ask_question_chat, inputs=[question_input, chatbot], outputs=[question_input, chatbot])

demo.launch()
