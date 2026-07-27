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

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

TOP_K = 5

# Modelos disponibles en tu API de Gemini
MODELOS_GEMINI = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest"
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
# Función: recuperar fragmentos desde FAISS
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
# Función: construir contexto para el LLM
# ============================================================

def construir_contexto(resultados):
    """
    Construye el bloque de contexto que será entregado al LLM.
    Incluye contenido y metadatos de cada fragmento recuperado.
    """

    contexto = ""

    for i, doc in enumerate(resultados, start=1):

        archivo = doc.metadata.get(
            "archivo",
            "Archivo no identificado"
        )

        categoria = doc.metadata.get(
            "categoria",
            "Categoría no identificada"
        )

        chunk = doc.metadata.get(
            "chunk_numero",
            "N/A"
        )

        contenido = doc.page_content

        contexto += (
            f"\n\n[Fragmento {i}]\n"
            f"Archivo: {archivo}\n"
            f"Categoría: {categoria}\n"
            f"Chunk: {chunk}\n"
            f"Contenido:\n"
            f"{contenido}\n"
        )

    return contexto.strip()


# ============================================================
# Función: obtener fuentes únicas
# ============================================================

def obtener_fuentes(resultados):
    """
    Extrae las fuentes utilizadas, evitando duplicados exactos.
    """

    fuentes = []

    for doc in resultados:

        archivo = doc.metadata.get(
            "archivo",
            "Archivo no identificado"
        )

        categoria = doc.metadata.get(
            "categoria",
            "Categoría no identificada"
        )

        chunk = doc.metadata.get(
            "chunk_numero",
            "N/A"
        )

        fuente = {
            "archivo": archivo,
            "categoria": categoria,
            "chunk": chunk
        }

        if fuente not in fuentes:
            fuentes.append(fuente)

    return fuentes


# ============================================================
# Función: construir prompt para Gemini
# ============================================================

def construir_prompt(pregunta, contexto):
    """
    Construye el prompt que se enviará al modelo generativo.
    Incluye reglas estrictas para reducir alucinaciones.
    """

    prompt = f"""
Eres un asistente corporativo especializado en normativa interna,
normativa externa y documentación de proyectos de Heavenward.

Tu tarea es responder preguntas de colaboradores utilizando
EXCLUSIVAMENTE el contexto recuperado desde los documentos disponibles.

REGLAS OBLIGATORIAS:
1. Responde solo con información contenida en el contexto proporcionado.
2. No inventes datos.
3. No uses conocimiento externo.
4. No extrapoles información normativa.
5. Si el contexto no contiene información suficiente, responde exactamente:
   "No encontré información suficiente en los documentos disponibles."
6. Si respondes, menciona claramente las fuentes utilizadas.
7. Usa un lenguaje claro, profesional y directo.
8. Si la pregunta está relacionada con normativa, responde con especial cuidado.
9. Si hay varias fuentes, integra la respuesta sin contradecir los documentos.
10. No menciones fragmentos que no aporten a la respuesta.

PREGUNTA DEL USUARIO:
{pregunta}

CONTEXTO RECUPERADO:
{contexto}

FORMATO DE RESPUESTA ESPERADO:

Respuesta:
[Responder de forma clara y directa.]

Fuentes utilizadas:
- [Nombre del archivo 1]
- [Nombre del archivo 2]

Si no existe información suficiente, responder:
"No encontré información suficiente en los documentos disponibles."

RESPUESTA:
"""

    return prompt


# ============================================================
# Función: generar respuesta con Gemini
# ============================================================

def generar_respuesta(prompt):
    """
    Genera una respuesta usando los modelos Gemini disponibles.
    Prueba varios modelos en orden hasta obtener una respuesta válida.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return (
            "No se encontró la variable GOOGLE_API_KEY en el archivo .env. "
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

            if respuesta and respuesta.text:
                return respuesta.text.strip()

        except Exception as error:
            ultimo_error = error

    return (
        "No fue posible generar una respuesta con Gemini. "
        f"Detalle técnico: {ultimo_error}"
    )


# ============================================================
# Función: mostrar fuentes utilizadas
# ============================================================

def mostrar_fuentes(fuentes):
    """
    Muestra las fuentes recuperadas por FAISS.
    """

    print("\n" + "=" * 80)
    print("FUENTES RECUPERADAS")
    print("=" * 80)

    if not fuentes:
        print("\nNo se recuperaron fuentes documentales.")
        return

    for i, fuente in enumerate(fuentes, start=1):

        print(f"\nFuente {i}")
        print(f"Archivo: {fuente['archivo']}")
        print(f"Categoría: {fuente['categoria']}")
        print(f"Chunk: {fuente['chunk']}")


# ============================================================
# Función: mostrar contexto recuperado
# ============================================================

def mostrar_contexto(contexto):
    """
    Muestra el contexto recuperado para validación técnica.
    Esta salida ayuda a auditar qué información se entregó al LLM.
    """

    print("\n" + "=" * 80)
    print("CONTEXTO RECUPERADO PARA EL LLM")
    print("=" * 80)

    print(contexto[:6000])

    if len(contexto) > 6000:
        print("\n[Contexto truncado en pantalla para facilitar lectura]")


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

    # --------------------------------------------------------
    # 1. Recuperar fragmentos relevantes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Construir contexto
    # --------------------------------------------------------

    contexto = construir_contexto(
        resultados
    )

    fuentes = obtener_fuentes(
        resultados
    )

    if not contexto.strip():
        print(
            "\nNo encontré información suficiente "
            "en los documentos disponibles."
        )
        return

    # --------------------------------------------------------
    # 3. Construir prompt
    # --------------------------------------------------------

    prompt = construir_prompt(
        pregunta,
        contexto
    )

    # --------------------------------------------------------
    # 4. Generar respuesta con Gemini
    # --------------------------------------------------------

    respuesta = generar_respuesta(
        prompt
    )

    # --------------------------------------------------------
    # 5. Mostrar respuesta final
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESPUESTA GENERADA")
    print("=" * 80)

    print("\n" + respuesta)

    # --------------------------------------------------------
    # 6. Mostrar fuentes
    # --------------------------------------------------------

    mostrar_fuentes(
        fuentes
    )

    # --------------------------------------------------------
    # 7. Mostrar contexto para auditoría
    # --------------------------------------------------------

    mostrar_contexto(
        contexto
    )

    print("\n" + "=" * 80)
    print("FIN")
    print("=" * 80)


if __name__ == "__main__":
    main()
