import requests
import chromadb
from pypdf import PdfReader
from pathlib import Path
import hashlib
import re

from tools import calculator


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"

GENERATION_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ============================================================
# CHROMA DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="multi_document_rag_v2"
)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
    )

    if not response.ok:

        print("\nOLLAMA EMBEDDING ERROR:")
        print(response.text)

        response.raise_for_status()

    return response.json()["embedding"]


# ============================================================
# FILE HASH
# ============================================================

def create_file_hash(file_path):

    md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(8192)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()


# ============================================================
# LOAD PDF PAGES
# ============================================================

def load_pdf_pages(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            text = text.strip()

            if text:

                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


# ============================================================
# CHUNK PAGE TEXT
# ============================================================

def chunk_page_text(
    text,
    max_chars=1000
):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if (
            len(current_chunk)
            + len(sentence)
            + 1
            <= max_chars
        ):

            if current_chunk:

                current_chunk += " "

            current_chunk += sentence

        else:

            if current_chunk:

                chunks.append(
                    current_chunk
                )

            current_chunk = sentence

    if current_chunk:

        chunks.append(
            current_chunk
        )

    return chunks


# ============================================================
# GET INDEXED FILE HASHES
# ============================================================

def get_indexed_file_hashes():

    try:

        result = collection.get(
            include=["metadatas"]
        )

        hashes = set()

        for metadata in result.get(
            "metadatas",
            []
        ):

            if (
                metadata
                and
                "file_hash" in metadata
            ):

                hashes.add(
                    metadata["file_hash"]
                )

        return hashes

    except Exception:

        return set()


# ============================================================
# INDEX DOCUMENTS
# ============================================================

def index_documents():

    print("\nChecking documents...\n")

    indexed_hashes = (
        get_indexed_file_hashes()
    )

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF files found in "
            "documents folder."
        )

        return

    for pdf_file in pdf_files:

        file_hash = create_file_hash(
            pdf_file
        )

        if file_hash in indexed_hashes:

            print(
                f"Skipping already indexed: "
                f"{pdf_file.name}"
            )

            continue

        print(
            f"Indexing: {pdf_file.name}"
        )

        pages = load_pdf_pages(
            pdf_file
        )

        for page_data in pages:

            page_number = page_data["page"]

            page_text = page_data["text"]

            chunks = chunk_page_text(
                page_text
            )

            for chunk_number, chunk in enumerate(
                chunks
            ):

                embedding = create_embedding(
                    chunk
                )

                document_id = (
                    f"{file_hash}_"
                    f"{page_number}_"
                    f"{chunk_number}"
                )

                collection.add(

                    ids=[
                        document_id
                    ],

                    embeddings=[
                        embedding
                    ],

                    documents=[
                        chunk
                    ],

                    metadatas=[{

                        "source":
                            pdf_file.name,

                        "page":
                            page_number,

                        "chunk":
                            chunk_number,

                        "file_hash":
                            file_hash
                    }]
                )

        print(
            f"Finished indexing: "
            f"{pdf_file.name}"
        )

    print(
        "\nDocument indexing complete.\n"
    )


# ============================================================
# CALCULATOR TOOL DEFINITION
# ============================================================

calculator_tool = {

    "type": "function",

    "function": {

        "name": "calculator",

        "description": (
            "Calculate mathematical "
            "expressions."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "expression": {

                    "type": "string",

                    "description": (
                        "Mathematical expression "
                        "such as 125 * 48."
                    )
                }
            },

            "required": [
                "expression"
            ]
        }
    }
}


# ============================================================
# DOCUMENT SEARCH
# ============================================================

def search_documents(question):

    print(
        "\n[RAG tool] Searching documents..."
    )

    question_embedding = create_embedding(
        question
    )

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=8
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    result_parts = []

    for i, document in enumerate(
        documents
    ):

        metadata = metadatas[i]

        distance = distances[i]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk = metadata.get(
            "chunk",
            "Unknown"
        )

        result_parts.append(

            f"""
[Source: {source}, Page {page}, Chunk {chunk}, Distance {distance:.4f}]

{document}
"""
        )

    if not result_parts:

        return (
            "No relevant document "
            "information was found."
        )

    return "\n".join(
        result_parts
    )


# ============================================================
# DOCUMENT SEARCH TOOL DEFINITION
# ============================================================

search_documents_tool = {

    "type": "function",

    "function": {

        "name":
            "search_documents",

        "description":
            (
                "Search the user's local PDF "
                "documents for information. "
                "Use this for questions about "
                "the user's resume, profile, "
                "skills, education, experience, "
                "projects, qualifications, "
                "certifications, or other "
                "stored documents."
            ),

        "parameters": {

            "type": "object",

            "properties": {

                "question": {

                    "type": "string",

                    "description":
                        (
                            "The question to search "
                            "for in the documents."
                        )
                }
            },

            "required": [
                "question"
            ]
        }
    }
}


# ============================================================
# AGENT TOOLS
# ============================================================

agent_tools = [

    calculator_tool,

    search_documents_tool
]


# ============================================================
# CALL OLLAMA CHAT
# ============================================================

def call_ollama_chat(
    messages,
    tools=None
):

    payload = {

        "model":
            GENERATION_MODEL,

        "messages":
            messages,

        "stream":
            False
    }

    if tools:

        payload["tools"] = tools

    response = requests.post(

        f"{OLLAMA_URL}/api/chat",

        json=payload
    )

    if not response.ok:

        print(
            "\nOLLAMA ERROR RESPONSE:"
        )

        print(
            response.text
        )

        response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Return ONLY the assistant message when there is no
    # tool call.
    #
    # But preserve the full response when tool calling
    # information is required.
    # --------------------------------------------------------

    message = data.get(
        "message",
        {}
    )

    tool_calls = message.get(
        "tool_calls"
    )

    if tool_calls:

        return data

    return message.get(
        "content",
        ""
    )


# ============================================================
# POLISHED ANSWER GENERATOR
# ============================================================

def generate_polished_answer(
    question,
    evidence,
    conversation_history=None
):

    if conversation_history is None:

        conversation_history = []

    history_text = ""

    for previous_user, previous_answer in (
        conversation_history[-5:]
    ):

        history_text += (

            f"\nUser: "
            f"{previous_user}\n"

            f"Assistant: "
            f"{previous_answer}\n"
        )

    system_prompt = f"""
You are a professional AI assistant.

Produce a clear, accurate, natural and
polished answer.

RULES:

1. Answer the user's question directly.

2. For document questions, use ONLY the
   provided document evidence.

3. Never invent information.

4. If the evidence does not contain the
   requested information, say:

   "I couldn't find that information in
   the provided documents."

5. Do not infer skills, qualifications,
   experience or abilities that are not
   explicitly supported by the evidence.

6. If the user asks for skills, identify
   only skills explicitly mentioned in
   the documents.

7. Do not treat random words in a document
   as user skills unless the document
   clearly presents them as skills.

8. Use headings or bullet points when
   useful.

9. Keep simple answers concise.

10. For multiple items, use bullet points.

11. Cite document information using:

    [Source: filename.pdf, Page X]

12. Never create fake citations.

13. Do not reveal hidden reasoning.

Conversation history:

{history_text}

Document evidence:

{evidence}
"""

    messages = [

        {
            "role":
                "system",

            "content":
                system_prompt
        },

        {
            "role":
                "user",

            "content":
                question
        }
    ]

    response = call_ollama_chat(
        messages
    )

    # --------------------------------------------------------
    # call_ollama_chat should normally return a string here.
    # This fallback keeps the function safe if a dictionary
    # somehow comes back.
    # --------------------------------------------------------

    if isinstance(
        response,
        dict
    ):

        return (
            response
            .get(
                "message",
                {}
            )
            .get(
                "content",
                ""
            )
        )

    return response