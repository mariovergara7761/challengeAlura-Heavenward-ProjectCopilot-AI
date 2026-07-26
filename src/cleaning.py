import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "src" / "output" / "documentos_extraidos.json"
OUTPUT_FILE = BASE_DIR / "src" / "output" / "documentos_limpios.json"

# ----------------------------------
# Función limpieza
# ----------------------------------

def limpiar_texto(texto):

    # eliminar saltos múltiples
    texto = re.sub(r"\n+", "\n", texto)

    # eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto)

    # eliminar textos repetitivos detectados
    patrones = [
        r"<< DOCUMENTO NO VIGENTE >>",
        r"Impreso el \d{2}-\d{2}-\d{4}",
        r"Página \d+ de \d+",
        r"Pág\. \d+ de \d+",
        r"Estado: En Construcción"
    ]

    for patron in patrones:
        texto = re.sub(patron, "", texto, flags=re.IGNORECASE)

    return texto.strip()

# ----------------------------------
# Cargar JSON
# ----------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documentos = json.load(f)

# ----------------------------------
# Limpiar documentos
# ----------------------------------

for documento in documentos:

    documento["texto"] = limpiar_texto(
        documento["texto"]
    )

# ----------------------------------
# Guardar resultado
# ----------------------------------

with open(
    OUTPUT_FILE,
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
print("✅ Limpieza completada")
print(f"✅ Archivo generado: {OUTPUT_FILE}")
