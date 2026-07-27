# ============================================================
# Heavenward ProjectCopilot AI
# Dockerfile
# ============================================================

FROM python:3.13-slim

# Evitar archivos pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Logs inmediatos
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && apt-get clean

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Copiar proyecto completo
COPY . .

# Puerto Streamlit
EXPOSE 8501

# Configuración Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Ejecutar aplicación
CMD ["streamlit", "run", "app.py"]
