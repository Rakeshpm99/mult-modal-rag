# Advanced Multimodal RAG Chatbot

This project is a sophisticated, multimodal Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, and Google's Gemini models. It can intelligently process and answer questions about complex PDF documents that contain both text and rich visual content like charts, graphs, and diagrams.

The application features an advanced on-demand analysis architecture to provide accurate, context-aware answers while efficiently managing API usage. The user interface is powered by Chainlit for a seamless and interactive experience.

## Key Features

* **True Multimodal Understanding:** Goes beyond simple OCR to interpret the content and meaning of charts, graphs, and diagrams using the Gemini 1.5 Flash vision model.

* **Hybrid Retrieval System:** Creates a unified vector store from both text chunks and discoverable "image placeholders" to ensure relevant context is never missed, whether it's in text or a graphic.

* **Efficient On-Demand Analysis:** To conserve API quotas and reduce initial processing time, image analysis is only performed when a user's query is relevant to a page containing an image.

* **Smart Caching & Interactive Workflow:** The application transparently informs the user when it finds relevant images and asks for permission before proceeding with analysis. Analyzed summaries are then cached for the session to prevent redundant API calls.

* **Built with LangChain:** Leverages the power and flexibility of the LangChain framework to structure the entire RAG pipeline, from document loading to the final QA chain.

## Architecture & Workflow

This chatbot uses an intelligent, hybrid on-demand architecture to solve the challenges of multimodal RAG:

* **Ingestion:** When a PDF is uploaded, the application extracts all text and creates Document chunks. Simultaneously, it identifies every image and creates a simple [Image Placeholder] document for each one, noting its page number.

* **Vector Store Creation:** A single ChromaDB vector store is created containing the embeddings for both the text chunks and the image placeholders. This makes images "discoverable" through semantic search.

* **Retrieval:** When a user asks a question, the retriever searches the hybrid vector store. The results may include both text chunks and image placeholders.

* **Interactive Analysis:** If any image placeholders are retrieved, the system identifies the new, un-analyzed pages and asks the user for permission to proceed with vision analysis.

* **On-Demand Summarization:** If permission is granted, the application calls the Gemini Vision API to generate a detailed summary for only the necessary images. These summaries are then cached.

* **Synthesis:** The retrieved text, along with any newly generated or cached image summaries, are combined into a rich context. This is passed to the final Gemini model to generate a comprehensive, synthesized answer.

## Tech Stack

* **Backend:** Python

* **AI Framework:** LangChain

* **LLMs:** Google Gemini 1.5 Flash (for chat and vision)

* **Embeddings:** Google models/embedding-001

* **UI:** Chainlit

* **Vector Database:** ChromaDB

* **PDF Processing:** PyMuPDF

## Setup & Usage
**1. Prerequisites**

* Python 3.8+

* Git

* A Google API Key with the Gemini API enabled.

**2. Installation**

Clone the repository and navigate into the project directory:

    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name

Create and activate a virtual environment:
    
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:

    pip install -r requirements.txt

**3. Configuration**

Create a file named .env in the root of the project directory and add your Google API Key:

    GOOGLE_API_KEY="YOUR_API_KEY_HERE"

Make sure you have a .gitignore file to prevent your .env file and other sensitive data from being committed.
**4. Running the Application**

Launch the Chainlit application from your terminal:

    chainlit run app.py -w

Open your web browser and navigate to http://localhost:8000 to start interacting with your multimodal chatbot.
