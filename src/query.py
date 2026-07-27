from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==================================
# Configuración
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    BASE_DIR /
    "vectorstore"
)

# ==================================
# Embeddings
# ==================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================
# Cargar índice FAISS
# ==================================

vectorstore = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

# ==================================
# Pregunta del usuario
# ==================================

pregunta = input(
    "\nEscribe tu consulta: "
)

# ==================================
# Recuperación semántica
# ==================================

resultados = vectorstore.similarity_search(
    pregunta,
    k=5
)

# ==================================
# Construcción del contexto
# ==================================

contexto = ""

for i, doc in enumerate(resultados, start=1):

    contexto += (
        f"\n\n### Fragmento {i}\n"
        f"Archivo: {doc.metadata.get('archivo')}\n"
        f"Categoría: {doc.metadata.get('categoria')}\n"
        f"Chunk: {doc.metadata.get('chunk_numero')}\n\n"
        f"{doc.page_content}\n"
    )

# ==================================
# Mostrar contexto consolidado
# ==================================

print("\n" + "=" * 80)
print("PREGUNTA")
print("=" * 80)

print(pregunta)

print("\n" + "=" * 80)
print("CONTEXTO RECUPERADO")
print("=" * 80)

print(contexto)

print("\n" + "=" * 80)
print("FIN DEL CONTEXTO")
print("=" * 80)
