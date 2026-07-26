import json
import os

from pathlib import Path

from langchain_core.documents import Document

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# ======================
# Configuración
# ======================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "src" /
    "output" /
    "chunks.json"
)

VECTORSTORE_DIR = (
    BASE_DIR /
    "vectorstore"
)

# ======================
# Leer chunks
# ======================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

# ======================
# Crear documentos
# ======================

documents = []

for chunk in chunks:

    doc = Document(
        page_content=chunk["texto"],
        metadata={
            "archivo":
                chunk["archivo"],

            "categoria":
                chunk["categoria"],

            "chunk_numero":
                chunk["chunk_numero"]
        }
    )

    documents.append(doc)

print(
    f"Documentos cargados: {len(documents)}"
)

# ======================
# Embeddings Gemini
# ======================

embeddings = (
    GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
)

# ======================
# Crear FAISS
# ======================

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

# ======================
# Guardar índice
# ======================

vectorstore.save_local(
    str(VECTORSTORE_DIR)
)

print()
print("✅ Embeddings generados")
print("✅ FAISS creado")
print(
    f"✅ Ubicación: {VECTORSTORE_DIR}"
)
