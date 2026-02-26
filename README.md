# Evaluación Ergonómica desde Foto (Oficina)

App web (Streamlit) para subir una foto lateral del puesto, detectar postura y descargar un informe PDF con recomendaciones estilo Prevención.

## 1) Desplegar en Streamlit Cloud (gratis)
1. Crea una cuenta en https://streamlit.io/ (Sign in)
2. Pulsa **"New app"** → Conecta con tu GitHub (te pedirá permiso una vez).
3. En GitHub, crea un repositorio (p.ej. `ergoprev-app`) y sube estos archivos: `app.py`, `ergonomics.py`, `report.py`, `requirements.txt`, `README.md`.
4. En Streamlit, elige ese repo y rama (main), y como **Main file path** pon `app.py`.
5. Pulsa **"Deploy"**. En ~1-3 min tendrás una URL pública del tipo `https://<tuapp>.streamlit.app`.

## 2) Uso
- Abre la URL.
- Sube una imagen (JPG/PNG) **preferiblemente lateral**.
- Revisa métricas y recomendaciones.
- Descarga el **PDF**.

## 3) Añadir como pestaña en Microsoft Teams
- En el canal de Teams → `+` (Agregar pestaña) → **Sitio web**.
- Pon un nombre (p.ej. “Evaluación Ergonómica”).
- Pega la URL de la app de Streamlit.
- Guardar. ¡Lista para todos!

## Notas
- Esta app no guarda datos. El PDF se genera en tu navegador.
- Para evaluación formal completa, compleméntalo con una visita o métodos como ROSA/REBA.
