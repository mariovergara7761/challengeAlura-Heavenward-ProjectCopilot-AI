import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# Configuración general
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

TOP_K = 5
TOP_K_CANDIDATOS = 8
SCORE_MAXIMO_REFERENCIAL = 1.5

MODELOS_GEMINI = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest"
]


# ============================================================
# Configuración visual de Streamlit
# ============================================================

st.set_page_config(
    page_title="Heavenward ProjectCopilot AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Carga de recursos con caché
# ============================================================

@st.cache_resource
def cargar_embeddings():
    """
    Carga el modelo local de embeddings.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


@st.cache_resource
def cargar_vectorstore():
    """
    Carga el índice FAISS generado previamente.
    """

    embeddings = cargar_embeddings()

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


@st.cache_resource
def cargar_cliente_gemini():
    """
    Carga el cliente Gemini usando la API Key del archivo .env.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    client = genai.Client(
        api_key=api_key
    )

    return client


# ============================================================
# Recuperación semántica
# ============================================================

def recuperar_contexto(pregunta):
    """
    Recupera fragmentos relevantes desde FAISS usando búsqueda semántica.
    """

    vectorstore = cargar_vectorstore()

    resultados_con_score = vectorstore.similarity_search_with_score(
        pregunta,
        k=TOP_K_CANDIDATOS
    )

    if not resultados_con_score:
        return []

    resultados_filtrados = [
        (doc, score)
        for doc, score in resultados_con_score
        if score <= SCORE_MAXIMO_REFERENCIAL
    ]

    if not resultados_filtrados:
        resultados_filtrados = resultados_con_score[:TOP_K]

    return resultados_filtrados[:TOP_K]


# ============================================================
# Construcción de contexto
# ============================================================

def construir_contexto(resultados_con_score):
    """
    Construye el contexto que será enviado al LLM.
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
# Fuentes documentales
# ============================================================

def obtener_fuentes(resultados_con_score):
    """
    Extrae fuentes únicas desde los resultados recuperados.
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
            "score": float(score)
        }

        if fuente not in fuentes:
            fuentes.append(fuente)

    return fuentes


# ============================================================
# Prompt para Gemini
# ============================================================

def construir_prompt(pregunta, contexto):
    """
    Construye el prompt principal para controlar alucinaciones.
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
# Generación de respuesta
# ============================================================

def generar_respuesta(prompt):
    """
    Genera una respuesta usando Gemini.
    Prueba varios modelos disponibles.
    """

    client = cargar_cliente_gemini()

    if client is None:
        return (
            "No se encontró la variable GOOGLE_API_KEY en el archivo .env. "
            "No fue posible generar una respuesta con Gemini."
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
# Pipeline principal del agente
# ============================================================

def responder_pregunta(pregunta):
    """
    Ejecuta el flujo completo:
    pregunta -> recuperación -> contexto -> Gemini -> respuesta.
    """

    resultados_con_score = recuperar_contexto(
        pregunta
    )

    if not resultados_con_score:
        return {
            "respuesta": "No encontré información suficiente en los documentos disponibles.",
            "fuentes": [],
            "contexto": ""
        }

    contexto = construir_contexto(
        resultados_con_score
    )

    if not contexto.strip():
        return {
            "respuesta": "No encontré información suficiente en los documentos disponibles.",
            "fuentes": [],
            "contexto": ""
        }

    fuentes = obtener_fuentes(
        resultados_con_score
    )

    prompt = construir_prompt(
        pregunta,
        contexto
    )

    respuesta = generar_respuesta(
        prompt
    )

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "contexto": contexto
    }


# ============================================================
# Estado de sesión
# ============================================================

if "historial" not in st.session_state:
    st.session_state.historial = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("⚙️ Configuración")

    st.markdown(
        """
        **Heavenward ProjectCopilot AI**

        Agente de consulta normativa y documental basado en RAG.

        **Estado actual:**
        - Documentos procesados
        - Embeddings generados
        - FAISS activo
        - Respuesta con Gemini
        """
    )

    st.divider()

    st.subheader("Base documental")

    st.markdown(
        """
        - 01 Normativa Interna
        - 02 Normativa Externa
        - 03 Proyectos
        """
    )

    st.divider()

    if st.button("🧹 Limpiar historial"):
        st.session_state.historial = []
        st.session_state.feedback = []
        st.rerun()

    st.divider()

    st.caption(
        "Este agente responde usando los documentos indexados disponibles."
    )


# ============================================================
# Encabezado principal
# ============================================================

st.title("🤖 Heavenward ProjectCopilot AI")

st.markdown(
    """
    Consulta normativa y documental basada en IA.

    Este asistente utiliza documentos internos, normativa externa y antecedentes
    de proyectos para responder preguntas con fuentes documentales.
    """
)

st.info(
    "Estás conversando con un agente de IA. "
    "Las respuestas deben validarse contra las fuentes citadas cuando se trate de información normativa o crítica."
)


# ============================================================
# Ejemplos de consulta
# ============================================================

with st.expander("💡 Ejemplos de preguntas"):

    st.markdown(
        """
        Puedes probar con preguntas como:

        - ¿Qué establece la POL-020 sobre información confidencial?
        - ¿Qué indica la NCh440-1 sobre requisitos de seguridad para ascensores eléctricos?
        - ¿Qué documentos hablan sobre entrega de instalación?
        - ¿Qué responsabilidades tiene el PCA según la política de confidencialidad?
        - ¿Qué información existe sobre insumos de SAP?
        """
    )


# ============================================================
# Mostrar historial
# ============================================================

for item in st.session_state.historial:

    with st.chat_message("user"):
        st.markdown(item["pregunta"])

    with st.chat_message("assistant"):
        st.markdown(item["respuesta"])

        if item.get("fuentes"):

            with st.expander("📚 Fuentes utilizadas"):

                for i, fuente in enumerate(item["fuentes"], start=1):

                    st.markdown(
                        f"""
                        **Fuente {i}**

                        - **Archivo:** {fuente["archivo"]}
                        - **Categoría:** {fuente["categoria"]}
                        - **Chunk:** {fuente["chunk"]}
                        - **Score FAISS:** {fuente["score"]}
                        """
                    )

        if item.get("contexto"):

            with st.expander("🔎 Ver contexto recuperado"):

                st.text(
                    item["contexto"][:6000]
                )

                if len(item["contexto"]) > 6000:
                    st.caption(
                        "Contexto truncado en pantalla para facilitar la revisión."
                    )


# ============================================================
# Entrada del usuario
# ============================================================

pregunta_usuario = st.chat_input(
    "Escribe tu consulta normativa o documental..."
)

if pregunta_usuario:

    with st.chat_message("user"):
        st.markdown(
            pregunta_usuario
        )

    with st.chat_message("assistant"):

        with st.spinner("Buscando información en la base documental..."):

            resultado = responder_pregunta(
                pregunta_usuario
            )

        st.markdown(
            resultado["respuesta"]
        )

        if resultado["fuentes"]:

            with st.expander("📚 Fuentes utilizadas"):

                for i, fuente in enumerate(resultado["fuentes"], start=1):

                    st.markdown(
                        f"""
                        **Fuente {i}**

                        - **Archivo:** {fuente["archivo"]}
                        - **Categoría:** {fuente["categoria"]}
                        - **Chunk:** {fuente["chunk"]}
                        - **Score FAISS:** {fuente["score"]}
                        """
                    )

        if resultado["contexto"]:

            with st.expander("🔎 Ver contexto recuperado"):

                st.text(
                    resultado["contexto"][:6000]
                )

                if len(resultado["contexto"]) > 6000:
                    st.caption(
                        "Contexto truncado en pantalla para facilitar la revisión."
                    )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Respuesta útil"):
                st.session_state.feedback.append(
                    {
                        "pregunta": pregunta_usuario,
                        "feedback": "positivo"
                    }
                )
                st.success("Feedback registrado.")

        with col2:
            if st.button("👎 Necesita mejora"):
                st.session_state.feedback.append(
                    {
                        "pregunta": pregunta_usuario,
                        "feedback": "negativo"
                    }
                )
                st.warning("Feedback registrado para mejora.")

    st.session_state.historial.append(
        {
            "pregunta": pregunta_usuario,
            "respuesta": resultado["respuesta"],
            "fuentes": resultado["fuentes"],
            "contexto": resultado["contexto"]
        }
    )
