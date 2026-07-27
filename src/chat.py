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
TOP_K_CANDIDATOS = 8

# Distancia máxima sugerida para FAISS.
# En FAISS con embeddings locales, menor puntaje = más parecido.
# Si el umbral deja pocos resultados, igual se usarán los mejores disponibles.
SCORE_MAXIMO_REFERENCIAL = 1.5

# Modelos reales disponibles en tu API Gemini.
# Importante: usar el prefijo "models/" porque así los devuelve tu API.
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
# Función: recuperar fragmentos relevantes con score
# ============================================================

def recuperar_contexto(pregunta, k_candidatos=TOP_K_CANDIDATOS, top_k=TOP_K):
    """
    Recupera fragmentos relevantes desde FAISS.

    Usa similarity_search_with_score para obtener:
    - documento recuperado
    - score de similitud/distancia

    Nota:
    En FAISS, normalmente un score menor indica mayor cercanía semántica.
    """

    resultados_con_score = vectorstore.similarity_search_with_score(
        pregunta,
        k=k_candidatos
    )

    if not resultados_con_score:
        return []

    # Filtrar resultados muy débiles según umbral referencial
    resultados_filtrados = [
        (doc, score)
        for doc, score in resultados_con_score
        if score <= SCORE_MAXIMO_REFERENCIAL
    ]

    # Si el filtro es demasiado estricto, usar los mejores resultados disponibles
    if not resultados_filtrados:
        resultados_filtrados = resultados_con_score[:top_k]

    return resultados_filtrados[:top_k]


# ============================================================
# Función: construir contexto para el LLM
# ============================================================

def construir_contexto(resultados_con_score):
    """
    Construye el bloque de contexto que será enviado al LLM.
    Incluye metadatos para que la respuesta pueda citar fuentes.
    """

    contexto = ""

    for i, (doc, score) in enumerate(resultados_con_score, start=1):

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
            f"Score FAISS: {score}\n"
            f"Contenido:\n"
            f"{contenido}\n"
        )

    return contexto.strip()


# ============================================================
# Función: obtener fuentes únicas
# ============================================================

def obtener_fuentes(resultados_con_score):
    """
    Extrae fuentes documentales utilizadas en la recuperación.
    Evita duplicados exactos.
    """

    fuentes = []

    for doc, score in resultados_con_score:

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
            "chunk": chunk,
            "score": score
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

    La instrucción principal es evitar alucinaciones:
    el modelo solo puede responder con base en el contexto recuperado.
    """

    prompt = f"""
Eres un asistente corporativo especializado en normativa interna,
normativa externa y documentación de proyectos de Heavenward.

Tu objetivo es responder preguntas de colaboradores usando EXCLUSIVAMENTE
el contexto recuperado desde los documentos disponibles.

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
10. No cites fuentes que no aporten directamente a la respuesta.

PREGUNTA DEL USUARIO:
{pregunta}

CONTEXTO RECUPERADO:
{contexto}

FORMATO DE RESPUESTA ESPERADO:

Respuesta:
[Respuesta clara, directa y basada solo en el contexto.]

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
    Genera una respuesta usando Gemini.

    Prueba los modelos disponibles en orden.
    Si un modelo falla, intenta con el siguiente.
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
# Función: mostrar fuentes
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
        print(f"Score FAISS: {fuente['score']}")


# ============================================================
# Función: mostrar contexto para auditoría
# ============================================================

def mostrar_contexto(contexto):
    """
    Muestra una vista parcial del contexto recuperado.
    Esto ayuda a auditar qué información fue entregada al LLM.
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

    resultados_con_score = recuperar_contexto(
        pregunta,
        k_candidatos=TOP_K_CANDIDATOS,
        top_k=TOP_K
    )

    if not resultados_con_score:
        print(
            "\nNo encontré información suficiente "
            "en los documentos disponibles."
        )
        return

    # --------------------------------------------------------
    # 2. Construir contexto
    # --------------------------------------------------------

    contexto = construir_contexto(
        resultados_con_score
    )

    if not contexto.strip():
        print(
            "\nNo encontré información suficiente "
            "en los documentos disponibles."
        )
        return

    fuentes = obtener_fuentes(
        resultados_con_score
    )

    # --------------------------------------------------------
    # 3. Construir prompt
    # --------------------------------------------------------

    prompt = construir_prompt(
        pregunta,
        contexto
    )

    # --------------------------------------------------------
    # 4. Generar respuesta
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
    # 6. Mostrar fuentes recuperadas
    # --------------------------------------------------------

    mostrar_fuentes(
        fuentes
    )

    # --------------------------------------------------------
    # 7. Mostrar contexto para auditoría técnica
    # --------------------------------------------------------

    mostrar_contexto(
        contexto
    )

    print("\n" + "=" * 80)
    print("FIN")
    print("=" * 80)


if __name__ == "__main__":
    main()
