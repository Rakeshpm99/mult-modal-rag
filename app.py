import os
import chainlit as cl
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
import time
import base64
from typing import Dict, List, Any

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file")

# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
CHAT_MODEL_NAME = "gemini-1.5-flash"
VISION_MODEL_NAME = "gemini-1.5-flash"

UPLOAD_DIR = "uploaded_docs"
CHROMA_DB_DIR = "chroma_db_multimodal_final_v3" # New DB for the final architecture

# --- Create Directories ---
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Initialize Models and Splitter ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
vision_llm = ChatGoogleGenerativeAI(model=VISION_MODEL_NAME, temperature=0.1, google_api_key=GOOGLE_API_KEY)
chat_llm = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, temperature=0.3, google_api_key=GOOGLE_API_KEY)

# --- Helper Functions ---

def process_pdf_with_text_tagging(pdf_path: str, file_name: str) -> (List[Document], Dict[int, List[str]]):
    """
    Processes a PDF using a text-tagging strategy.
    - First, identifies all pages that contain images.
    - Extracts text chunks and appends a special tag to any chunk from a page with images.
    - Stores the actual image data (base64) in a separate dictionary for on-demand use.
    """
    image_store = {}
    pages_with_images = set()

    # 1. First pass: Identify pages with images and store the image data
    print("First pass: Identifying pages with images...")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page_images_base64 = []
        for img in doc.load_page(page_num).get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            page_images_base64.append(img_base64)
        
        if page_images_base64:
            pages_with_images.add(page_num + 1)
            image_store[page_num + 1] = page_images_base64
    doc.close()
    print(f"Found images on pages: {sorted(list(pages_with_images))}")

    # 2. Second pass: Extract text and tag the chunks
    print("Second pass: Extracting and tagging text chunks...")
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    
    text_chunks = text_splitter.split_documents(pdf_docs)
    
    for chunk in text_chunks:
        chunk.metadata["source"] = file_name
        page_number = chunk.metadata.get("page")
        if page_number and page_number in pages_with_images:
            # Append the special tag to the content
            chunk.page_content += "\n\n[Context: This page also contains a visual element like a chart or diagram.]"
    
    print(f"Created and tagged {len(text_chunks)} text chunks.")
    return text_chunks, image_store

async def summarize_images_on_demand(images_base64: List[str], page_number: int) -> str:
    """
    Generates a summary for a list of images from a specific page when they are needed.
    """
    summaries = []
    print(f"Analyzing images on page {page_number}...")
    try:
        for i, img_b64 in enumerate(images_base64):
            msg = await vision_llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"This image is from page {page_number}. Describe it in detail. If it's a chart, graph, or diagram, explain its title, axes, data, and the main conclusion it presents."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ]
                    )
                ]
            )
            summaries.append(msg.content)
            await cl.sleep(1.5) # Safety delay
    except Exception as e:
        error_message = f"Error generating summary for images on page {page_number}: {e}"
        print(error_message)
        await cl.Message(content=error_message).send()
        return ""
    
    return "\n\n".join(summaries)

# --- Chainlit App Events ---

@cl.on_chat_start
async def start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF document to begin.",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=300
        ).send()

    uploaded_file = files[0]
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    with open(uploaded_file.path, "rb") as src, open(file_path, "wb") as dst:
        dst.write(src.read())
    
    msg = cl.Message(content=f"Processing `{uploaded_file.name}`... This will be quick!")
    await msg.send()

    # Process PDF using the new text-tagging method
    tagged_text_chunks, image_store = await cl.make_async(process_pdf_with_text_tagging)(file_path, uploaded_file.name)
    
    if not tagged_text_chunks:
        await cl.Message(content="Could not extract any content from the document.").send()
        return

    print("Creating vector store from tagged text chunks...")
    vectorstore = await cl.make_async(Chroma.from_documents)(
        documents=tagged_text_chunks,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_DIR
    )
    print("Vector store created.")

    cl.user_session.set("retriever", vectorstore.as_retriever(search_kwargs={"k": 7}))
    cl.user_session.set("image_store", image_store)
    cl.user_session.set("summaries_cache", {})

    msg.content = f"✅ Processing complete! You can now ask any questions about `{uploaded_file.name}`."
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    image_store = cl.user_session.get("image_store")
    summaries_cache = cl.user_session.get("summaries_cache")

    if not retriever:
        await cl.Message(content="The document has not been processed yet.").send()
        return

    # 1. Retrieve relevant text chunks (some will be tagged)
    retrieved_docs = await retriever.ainvoke(message.content)
    
    text_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. Find pages with images that need to be analyzed based on the retrieved docs
    relevant_pages = sorted(list(set(doc.metadata.get("page") for doc in retrieved_docs if doc.metadata.get("page") in image_store)))
    pages_to_analyze = [p for p in relevant_pages if p not in summaries_cache]

    # 3. Ask user for permission to analyze new images
    if pages_to_analyze:
        # FIX: Added payload={} to satisfy a requirement in certain versions of Chainlit.
        action_buttons = [
            cl.Action(name="analyze", value="yes", label="Yes, analyze them", payload={}),
            cl.Action(name="skip", value="no", label="No, answer without them", payload={}),
        ]
        res = await cl.AskActionMessage(
            content=f"I found potentially relevant images on pages: {pages_to_analyze}. Would you like me to analyze them? This may use API credits.",
            actions=action_buttons,
            timeout=60
        ).send()

        if res and res.get("value") == "yes":
            for page_num in pages_to_analyze:
                summary = await summarize_images_on_demand(image_store[page_num], page_num)
                if summary:
                    summaries_cache[page_num] = summary

    # 4. Assemble the final context from text and all relevant cached summaries
    image_context = ""
    for page_num in relevant_pages:
        if page_num in summaries_cache:
            image_context += f"\n\n--- Summary of Image(s) on Page {page_num} ---\n{summaries_cache[page_num]}"

    # 5. Construct the final prompt and generate the answer
    prompt_template = """You are an expert AI assistant. Your task is to answer the user's question based on the provided context.
The context contains two parts: text excerpts and summaries of relevant images.
Some text excerpts may contain a note like '[Context: This page also contains a visual element...]'. Use this as a strong hint to look at the corresponding image summary.
Synthesize information from both sources to provide a comprehensive response.

---
Text Context:
{text_context}
---
Image Summary Context:
{image_context}
---

Question:
{question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["text_context", "image_context", "question"]
    )
    
    chain = PROMPT | chat_llm
    
    msg = cl.Message(content="")
    await msg.send()

    final_context = {
        "text_context": text_context,
        "image_context": image_context if image_context else "No relevant images were analyzed for this query.",
        "question": message.content
    }

    async for chunk in chain.astream(final_context):
        await msg.stream_token(chunk.content)
    
    await msg.update()
