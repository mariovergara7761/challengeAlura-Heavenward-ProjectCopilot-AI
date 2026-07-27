import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# Configuración general
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    BASE_DIR /
    "vectorstore"
)

TOP_K = 5

MODELOS_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]


# ============================================================
# Cargar embeddings locales
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Cargar índice FAISS
# ============================================================

vectorstore = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# Función: recuperar documentos relevantes
# ============================================================

def recuperar_contexto(pregunta, k=TOP_K):
    """
    Recupera los fragmentos más relevantes desde FAISS.
    """

    resultados = vectorstore.similarity_search(
        pregunta,
        k=k
    )

    return resultados


# ============================================================
# Función: ensamblar contexto para el LLM
# ============================================================

def construir_contexto(resultados):
    """
    Construye un bloque de contexto con metadatos de fuente.
    """

    contexto = ""

    for i, doc in enumerate(resultados, start=1):

        archivo = doc.metadata.get("archivo", "Fuente no identificada")
        categoria = doc.metadata.get("categoria", "Categoría no identificada")
        chunk = doc.metadata.get("chunk_numero", "N/A")

        contexto += (
            f"\n\n[Fragmento {i}]\n"
            f"Archivo: {archivo}\n"
            f"Categoría: {categoria}\n"
            f"Chunk: {chunk}\n"
            f"Contenido:\n"
            f"{doc.page_content}\n"
        )

    return contexto.strip()


# ============================================================
# Función: obtener fuentes únicas
# ============================================================

def obtener_fuentes(resultados):
    """
    Extrae las fuentes documentales utilizadas en la respuesta.
    """

    fuentes = []

    for doc in resultados:

        archivo = doc.metadata.get("archivo", "Fuente no identificada")
        categoria = doc.metadata.get("categoria", "Categoría no identificada")
        chunk = doc.metadata.get("chunk_numero", "N/A")

        fuente = {
            "archivo": archivo,
            "categoria": categoria,
            "chunk": chunk
        }

        if fuente not in fuentes:
            fuentes.append(fuente)

    return fuentes


# ============================================================
# Función: construir prompt
# ============================================================

def construir_prompt(pregunta, contexto):
    """
    Construye el prompt que será enviado al modelo generativo.
    """

    prompt = f"""
Eres un asistente corporativo especializado en normativa interna,
normativa externa y documentación de proyectos de Heavenward.

Tu tarea es responder preguntas usando EXCLUSIVAMENTE el contexto proporcionado.

Reglas obligatorias:
1. Responde solo con información contenida en el contexto.
2. No inventes datos.
3. No uses conocimiento externo.
4. Si el contexto no contiene información suficiente, responde:
   "No encontré información suficiente en los documentos disponibles."
5. Si respondes, menciona las fuentes utilizadas al final.
6. Usa un lenguaje claro, profesional y directo.
7. Si la pregunta se relaciona con normativa, responde con especial cuidado y sin extrapolar.

Pregunta del usuario:
{pregunta}

Contexto recuperado:
{contexto}

Respuesta:
"""

    return prompt


# ============================================================
# Función: generar respuesta con Gemini
# ============================================================

def generar_respuesta(prompt):
    """
    Genera una respuesta usando Gemini.
    Prueba varios modelos disponibles en caso de error de modelo.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return (
            "No se encontró GOOGLE_API_KEY en el archivo .env. "
            "No fue posible generar una respuesta con Gemini."
        )

    client = genai.Client(
        api_key=api_key
    )

    ultimo_error = None

    for modelo in MODELOS_GEMINI:

        try:
            respuesta = client.models.generate_content(
                model=modelo,
                contents=prompt
            )

            return respuesta.text

        except Exception as error:
            ultimo_error = error

    return (
        "No fue posible generar una respuesta con Gemini. "
        f"Detalle técnico: {ultimo_error}"
    )


# ============================================================
# Función: mostrar fuentes
# ============================================================

def mostrar_fuentes(fuentes):
    """
    Imprime las fuentes utilizadas por la recuperación.
    """

    print("\n" + "=" * 80)
    print("FUENTES UTILIZADAS")
    print("=" * 80)

    for i, fuente in enumerate(fuentes, start=1):

        print(f"\nFuente {i}")
        print(f"Archivo: {fuente['archivo']}")
        print(f"Categoría: {fuente['categoria']}")
        print(f"Chunk: {fuente['chunk']}")


# ============================================================
# Programa principal
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("HEAVENWARD PROJECTCOPILOT AI")
    print("Consulta normativa y documental")
    print("=" * 80)

    pregunta = input(
        "\nEscribe tu consulta: "
    ).strip()

    if not pregunta:
        print("\nNo ingresaste una consulta.")
        return

    # ------------------------------------------
    # Recuperar fragmentos relevantes
    # ------------------------------------------

    resultados = recuperar_contexto(
        pregunta,
        k=TOP_K
    )

    if not resultados:
        print(
            "\nNo encontré información suficiente "
            "en los documentos disponibles."
        )
        return

    # ------------------------------------------
    # Construir contexto
    # ------------------------------------------

    contexto = construir_contexto(
        resultados
    )

    fuentes = obtener_fuentes(
        resultados
    )

    # ------------------------------------------
    # Construir prompt
    # ------------------------------------------

    prompt = construir_prompt(
        pregunta,
        contexto
    )

    # ------------------------------------------
    # Generar respuesta
    # ------------------------------------------

    respuesta = generar_respuesta(
        prompt
    )

    # ------------------------------------------
    # Mostrar respuesta final
    # ------------------------------------------

    print("\n" + "=" * 80)
    print("RESPUESTA")
    print("=" * 80)

    print(
        respuesta
    )

    # ------------------------------------------
    # Mostrar fuentes utilizadas
    # ------------------------------------------

    mostrar_fuentes(
        fuentes
    )

    print("\n" + "=" * 80)
    print("FIN")
    print("=" * 80)


if __name__ == "__main__":
    main()
