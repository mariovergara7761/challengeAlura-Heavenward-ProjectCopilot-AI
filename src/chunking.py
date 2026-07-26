import json
from pathlib import Path

# ==================================
# Configuración
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "src" /
    "output" /
    "documentos_limpios.json"
)

OUTPUT_FILE = (
    BASE_DIR /
    "src" /
    "output" /
    "chunks.json"
)

# ==================================
# Parámetros
# ==================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==================================
# Función chunking
# ==================================

def crear_chunks(texto):

    chunks = []

    inicio = 0

    while inicio < len(texto):

        fin = inicio + CHUNK_SIZE

        chunk = texto[inicio:fin]

        chunks.append(chunk)

        inicio += (
            CHUNK_SIZE -
            CHUNK_OVERLAP
        )

    return chunks

# ==================================
# Cargar documentos
# ==================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    documentos = json.load(f)

# ==================================
# Generar chunks
# ==================================

resultado = []

for documento in documentos:

    chunks = crear_chunks(
        documento["texto"]
    )

    for numero, chunk in enumerate(
        chunks,
        start=1
    ):

        resultado.append(
            {
                "chunk_id":
                    f"{documento['archivo']}_{numero}",

                "archivo":
                    documento["archivo"],

                "categoria":
                    documento["categoria"],

                "chunk_numero":
                    numero,

                "texto":
                    chunk
            }
        )

# ==================================
# Guardar
# ==================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resultado,
        f,
        ensure_ascii=False,
        indent=4
    )

print()
print("✅ Chunking completado")
print(
    f"✅ Chunks generados: {len(resultado)}"
)
print(
    f"✅ Archivo: {OUTPUT_FILE}"
)
