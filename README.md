# Evaluación Ergonómica desde Foto (YOLOv8-pose)

App web (Streamlit) para subir una foto lateral del puesto, detectar postura con **YOLOv8‑pose** (Ultralytics) y descargar un informe PDF con recomendaciones estilo Prevención.

## 1) Despliegue en Streamlit Cloud
1. Crea un repositorio en GitHub con estos archivos (`app.py`, `ergonomics.py`, `report.py`, `requirements.txt`, `runtime.txt`, `README.md`).
2. Entra en https://streamlit.io → *New app* → selecciona tu repo.
3. **Main file path**: `app.py` → *Deploy*.
4. Primer uso: el modelo `yolov8n-pose.pt` se descargará automáticamente (unos segundos).

## 2) Uso
- Abre la URL de la app.
- Sube una imagen (JPG/PNG) preferiblemente **lateral**.
- Revisa métricas y recomendaciones.
- Descarga el **PDF**.

## Notas
- YOLOv8‑pose usa el formato COCO de 17 puntos. La precisión mejora con fotos laterales y buena iluminación.
- El análisis es aproximado y no sustituye una evaluación in situ.
