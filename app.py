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
CHROMA_DB_DIR = "chroma_db_multimodal_final" # New DB for the final architecture

# --- Create Directories ---
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Initialize Models and Splitter ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
vision_llm = ChatGoogleGenerativeAI(model=VISION_MODEL_NAME, temperature=0.1, google_api_key=GOOGLE_API_KEY)
chat_llm = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, temperature=0.3, google_api_key=GOOGLE_API_KEY)

# --- Helper Functions ---

def process_pdf_for_hybrid_retrieval(pdf_path: str, file_name: str) -> (List[Document], Dict[int, List[str]]):
    """
    Processes a PDF for a hybrid on-demand retrieval strategy.
    - Extracts and chunks text.
    - Creates simple placeholder documents for images to make them discoverable.
    - Stores the actual image data (base64) in a separate dictionary for on-demand use.
    """
    all_docs_for_vectorstore = []
    image_store = {}

    # 1. Extract and chunk text
    print(f"Loading text from PDF: {pdf_path}")
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    
    text_chunks = text_splitter.split_documents(pdf_docs)
    for chunk in text_chunks:
        chunk.metadata["source"] = file_name
    all_docs_for_vectorstore.extend(text_chunks)
    print(f"Created {len(text_chunks)} text chunks.")

    # 2. Extract images, store them, and create simple placeholders
    print("Extracting images and creating placeholders...")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page_images_base64 = []
        
        for img_index, img in enumerate(doc.load_page(page_num).get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            page_images_base64.append(img_base64)
            
            # Create a simple, structured placeholder document for each image
            placeholder_doc = Document(
                page_content=f"[Image Placeholder: An image, chart, or graphic is on this page. Query the image directly to understand its content.]",
                metadata={"source": file_name, "page": page_num + 1, "type": "image_placeholder"}
            )
            all_docs_for_vectorstore.append(placeholder_doc)
        
        if page_images_base64:
            image_store[page_num + 1] = page_images_base64
            
    doc.close()
    print(f"Stored images from {len(image_store)} pages and created placeholders.")
    return all_docs_for_vectorstore, image_store

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
            await cl.sleep(1) # Small safety delay between API calls
    except Exception as e:
        print(f"Error generating summary for images on page {page_number}: {e}")
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

    # Process PDF for hybrid retrieval
    all_docs, image_store = await cl.make_async(process_pdf_for_hybrid_retrieval)(file_path, uploaded_file.name)
    
    if not all_docs:
        await cl.Message(content="Could not extract any content from the document.").send()
        return

    # Create a single vector store from text chunks AND image placeholders
    print("Creating hybrid vector store...")
    vectorstore = await cl.make_async(Chroma.from_documents)(
        documents=all_docs,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_DIR
    )
    print("Vector store created.")

    cl.user_session.set("retriever", vectorstore.as_retriever(search_kwargs={"k": 10}))
    cl.user_session.set("image_store", image_store)

    msg.content = f"✅ Processing complete! You can now ask any questions about `{uploaded_file.name}`."
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    image_store = cl.user_session.get("image_store")

    if not retriever:
        await cl.Message(content="The document has not been processed yet.").send()
        return

    # 1. Retrieve relevant documents (will include text and image placeholders)
    retrieved_docs = await retriever.ainvoke(message.content)
    
    # Separate text from image placeholders
    text_context_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") != "image_placeholder"]
    image_placeholder_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image_placeholder"]
    
    text_context = "\n\n".join([doc.page_content for doc in text_context_docs])
    
    # 2. Find relevant pages from placeholders and summarize images on-demand
    relevant_pages = sorted(list(set(doc.metadata.get("page") for doc in image_placeholder_docs)))
    image_context = ""
    
    if relevant_pages:
        await cl.Message(content=f"Found potentially relevant images on pages: {relevant_pages}. Analyzing them now...").send()
        for page_num in relevant_pages:
            if page_num in image_store:
                image_summaries = await summarize_images_on_demand(image_store[page_num], page_num)
                if image_summaries:
                    image_context += f"\n\n--- Summary of Image(s) on Page {page_num} ---\n{image_summaries}"

    # 3. Construct the final prompt
    prompt_template = """You are an expert AI assistant. Your task is to answer the user's question based on the provided context.
The context contains two parts: text excerpts and, if available, summaries of relevant images that were just generated.
Synthesize information from both sources to provide a comprehensive response.
If the question refers to a visual element (chart, graph, diagram), prioritize the image summary context.

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
        "text_context": text_context if text_context else "No relevant text found.",
        "image_context": image_context if image_context else "No relevant images were found or analyzed for this query.",
        "question": message.content
    }

    async for chunk in chain.astream(final_context):
        await msg.stream_token(chunk.content)
    
    await msg.update()
