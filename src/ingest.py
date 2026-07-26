from pathlib import Path
from pypdf import PdfReader
import json

# ==================================
# Configuración
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "src" / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==================================
# Lectura PDF
# ==================================

def extraer_texto_pdf(pdf_path):

    texto = ""

    try:

        reader = PdfReader(pdf_path)

        for numero_pagina, pagina in enumerate(reader.pages):

            pagina_texto = pagina.extract_text()

            if pagina_texto:

                texto += pagina_texto + "\n"

        return texto

    except Exception as error:

        print(
            f"❌ Error leyendo {pdf_path.name}: {error}"
        )

        return ""

# ==================================
# Procesamiento
# ==================================

documentos = []

for archivo in DATA_DIR.rglob("*.pdf"):

    print(f"📄 Leyendo: {archivo.name}")

    texto = extraer_texto_pdf(archivo)

    documentos.append(
        {
            "archivo": archivo.name,
            "categoria": archivo.parent.name,
            "texto": texto
        }
    )

# ==================================
# Exportación JSON
# ==================================

salida = OUTPUT_DIR / "documentos_extraidos.json"

with open(
    salida,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documentos,
        f,
        ensure_ascii=False,
        indent=4
    )

print()
print("✅ Proceso completado")
print(f"✅ Documentos procesados: {len(documentos)}")
print(f"✅ Archivo generado: {salida}")
