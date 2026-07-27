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

Usuario
↓
Streamlit
↓
chat.py
↓
FAISS
↓
Contexto
↓
Gemini
↓
Respuesta
↓
Fuentes

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

### Streamlit

Agregar captura de Streamlit funcionando.

### Docker

Agregar captura de imagen Docker creada.

### Contenedor

Agregar captura del contenedor ejecutándose.

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