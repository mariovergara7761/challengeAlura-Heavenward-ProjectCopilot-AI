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
# Cargar FAISS
# ==================================

vectorstore = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

# ==================================
# Pregunta usuario
# ==================================

pregunta = input(
    "\nEscribe tu consulta: "
)

# ==================================
# Búsqueda semántica
# ==================================

resultados = vectorstore.similarity_search(
    pregunta,
    k=5
)

# ==================================
# Mostrar resultados
# ==================================

print("\n" + "=" * 80)
print("RESULTADOS")
print("=" * 80)

for i, doc in enumerate(resultados, start=1):

    print(f"\nResultado {i}")
    print("-" * 80)

    print(
        f"Archivo: {doc.metadata.get('archivo')}"
    )

    print(
        f"Categoría: {doc.metadata.get('categoria')}"
    )

    print(
        f"Chunk: {doc.metadata.get('chunk_numero')}"
    )

    print("\nTexto:")

    print(doc.page_content[:1000])

    print("\n" + "-" * 80)
