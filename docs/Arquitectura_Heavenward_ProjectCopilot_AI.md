# Heavenward ProjectCopilot AI

## Resumen Ejecutivo

Heavenward ProjectCopilot AI es un agente RAG (Retrieval Augmented Generation)
desarrollado para responder consultas sobre normativa interna, normativa externa
y documentación de proyectos de Heavenward Ascensores S.A.

## Objetivos

- Centralizar conocimiento documental.
- Reducir tiempos de búsqueda.
- Entregar respuestas con fuentes verificables.
- Minimizar alucinaciones mediante recuperación documental.

## Arquitectura General

┌─────────────────────────────┐
│      Usuario Final          │
│  (Colaborador Heavenward)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Streamlit (app.py)     │
│ Interfaz Web Conversacional │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         chat.py             │
│ Orquestación del Agente IA  │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│   query.py   │  │ Gemini 2.5   │
│ Recuperación │  │ Flash        │
│ Semántica    │  │ Respuestas   │
└───────┬──────┘  └──────────────┘
        │
        ▼
┌─────────────────────────────┐
│       FAISS VectorStore     │
│ index.faiss / index.pkl     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Embeddings MiniLM-L6-v2     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Chunks Documentales         │
│ chunking.py                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Documentos PDF              │
│ Normativa / Proyectos       │
└─────────────────────────────┘

## Tecnologías utilizadas

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Gemini 2.5 Flash
- Docker
- GitHub

## Flujo del Sistema

Función de cada componente
* ingest.py: Extrae texto desde PDF
* cleaning.py: Limpia y normaliza contenido
* chunking.py: Divide texto en fragmentos
* embeddings.py: Genera vectores semánticos
* FAISS: Almacena y busca embeddings

### 1. Ingesta documental

src/ingest.py

- Lectura de PDFs
- Extracción de texto

### 2. Limpieza

src/cleaning.py

- Normalización del texto
- Eliminación de caracteres no deseados

### 3. Chunking

src/chunking.py

Configuración:

- CHUNK_SIZE = 2500
- CHUNK_OVERLAP = 300

### 4. Embeddings

src/embeddings.py

Modelo:

sentence-transformers/all-MiniLM-L6-v2

### 5. Vector Store

FAISS

Archivos:

- vectorstore/index.faiss
- vectorstore/index.pkl

### 6. Recuperación Semántica

src/query.py

Pregunta
↓
Embeddings
↓
FAISS
↓
Top K resultados

### 7. Generación de Respuestas

src/chat.py

Pregunta
↓
Contexto
↓
Gemini
↓
Respuesta

### 8. Interfaz

app.py

Funcionalidades:

- Chat conversacional
- Historial
- Fuentes documentales
- Feedback
- Auditoría de contexto

## Dockerización

Dockerfile validado exitosamente.

Comandos principales:

docker build -t heavenward-projectcopilot-ai .

docker run -p 8501:8501 --env-file .env heavenward-projectcopilot-ai

## Evidencias

### Interfaz principal del sistema

Pantalla inicial de Heavenward ProjectCopilot AI desarrollada en Streamlit.
La interfaz permite realizar consultas sobre normativa interna,
normativa externa y documentación de proyectos.

### Procesamiento de consultas

Ejemplo de una consulta realizada por el usuario.
El sistema recupera información desde la base documental mediante FAISS
y construye el contexto necesario para la generación de respuestas.

### Respuesta generada por el Agente IA

Ejemplo de respuesta obtenida mediante recuperación semántica y generación
de contenido con Gemini 2.5 Flash.

La respuesta incluye:

- Información contextual.
- Fuentes documentales utilizadas.
- Trazabilidad de la información.
- Mecanismos de auditoría del contexto recuperado.

## Arquitectura OCI Objetivo

GitHub
↓
OCI Compute
↓
Docker
↓
Streamlit
↓
Usuarios

Servicios OCI considerados:

- OCI Compute
- OCI Object Storage
- OCI Vault
- OCI DevOps

## Estado del Proyecto

✅ Colecta documental

✅ Extracción

✅ Chunking

✅ Embeddings

✅ FAISS

✅ Recuperación RAG

✅ Generación de respuestas

✅ Streamlit

✅ Docker

🔄 OCI Compute

## Conclusiones

El proyecto implementa un agente RAG funcional capaz de consultar
documentación corporativa, recuperar contexto relevante y generar
respuestas fundamentadas utilizando Gemini.
