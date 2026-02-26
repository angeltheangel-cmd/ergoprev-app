# app.py
import io
import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from ergonomics import analyze_posture, draw_landmarks_bgr
from report import build_pdf_report

st.set_page_config(
    page_title="Evaluación Ergonómica - Puesto de Oficina",
    page_icon="🪑",
    layout="centered",
)

st.title("🪑 Evaluación Ergonómica desde una Foto (Oficina)")
st.caption("Sube una foto lateral de una persona sentada. Obtén un informe PDF con recomendaciones estilo Prevención.")

with st.expander("📸 Requisitos de la foto (consejos rápidos)", expanded=False):
    st.write(
        "- Vista **lateral** preferiblemente (a la altura del hombro).\n"
        "- Buena iluminación y postura habitual de trabajo (sin posar).\n"
        "- Silla, mesa y pantalla visibles si es posible.\n"
        "- Evitar ropa muy holgada que tape hombros/codo/cadera."
    )

uploaded = st.file_uploader("Sube una foto (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
logo_text = col1.text_input("Nombre/Logo para el informe (opcional)", value="Departamento de Prevención")
case_ref = col2.text_input("Identificador de evaluación (opcional)", value="Caso-001")

generate_btn = st.button("🔍 Analizar y generar informe")

if uploaded and generate_btn:
    # Cargar imagen
    image = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Analizando postura..."):
        try:
            result = analyze_posture(img_bgr)
        except Exception as e:
            st.error(f"No se pudo analizar la imagen: {e}")
            st.stop()

    if not result or not result.get("metrics"):
        st.error("No se detectó la postura. Prueba con una vista más lateral o mejor iluminación.")
        st.stop()

    metrics = result["metrics"]
    recommendations = result["recommendations"]
    overlay_bgr = draw_landmarks_bgr(img_bgr, result.get("landmarks_xy"))

    # Mostrar resultados
    st.subheader("🧭 Métricas estimadas (grados aproximados)")
    mcols = st.columns(2)
    left_keys = ["neck_flexion_deg", "trunk_flexion_deg", "hip_right_deg"]
    right_keys = ["knee_right_deg", "elbow_right_deg", "wrist_ext_right_deg"]
    for k in left_keys:
        if k in metrics and metrics[k] is not None:
            mcols[0].metric(k, f"{metrics[k]:.1f}°")
    for k in right_keys:
        if k in metrics and metrics[k] is not None:
            mcols[1].metric(k, f"{metrics[k]:.1f}°")

    st.subheader("🖼️ Detección de pose")
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    st.image(overlay_rgb, caption="Puntos corporales detectados", use_column_width=True)

    st.subheader("✅ Recomendaciones priorizadas")
    if recommendations:
        for title, text in recommendations:
            st.markdown(f"**{title}** — {text}")
    else:
        st.info("No se detectan desviaciones relevantes respecto a los rangos recomendados.")

    # Crear PDF en memoria
    st.subheader("📄 Informe PDF")
    pdf_bytes = build_pdf_report(
        original_rgb=np.array(image),
        overlay_rgb=overlay_rgb,
        metrics=metrics,
        recommendations=recommendations,
        logo_text=logo_text,
        case_ref=case_ref,
    )
    ts = int(time.time())
    file_name = f"informe_ergonomico_{case_ref or 'evaluacion'}_{ts}.pdf"
    st.download_button(
        label="⬇️ Descargar informe PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )

    st.success("Informe generado. Puedes descargar el PDF y compartirlo.")
elif not uploaded:
    st.info("Sube una imagen para empezar.")
