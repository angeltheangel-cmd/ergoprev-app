# report.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
import io
import tempfile
from PIL import Image


def _wrap_text_lines(text, width=90):
    words = text.split()
    line, out = "", []
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    return out


def build_pdf_report(original_rgb, overlay_rgb, metrics, recommendations, logo_text="", case_ref=""):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 2*cm

    # ---------- Portada ----------
    c.setFillColor(colors.HexColor("#0E7490"))
    c.rect(0, H-3*cm, W, 3*cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, H - 2*cm, "Informe de Evaluación Ergonómica")
    c.setFont("Helvetica", 11)
    c.drawString(margin, H - 2.8*cm, "Puesto de trabajo con ordenador (análisis a partir de imagen única)")

    c.setFillColor(colors.black)
    y = H - 4.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Identificación")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Entidad/Logo: {logo_text or '-'}")
    y -= 0.45*cm
    c.drawString(margin, y, f"Referencia de evaluación: {case_ref or '-'}")
    y -= 0.45*cm
    c.drawString(margin, y, "Método: YOLOv8-pose (Ultralytics) + reglas de ergonomía de oficina")
    y -= 0.45*cm
    c.drawString(margin, y, "Limitaciones: Monocular 2D, estimaciones angulares aproximadas, sin calibración de profundidad.")
    y -= 0.9*cm

    # Imagen resumen (overlay)
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(overlay_rgb).save(tmp.name, format="PNG")
            c.drawImage(tmp.name, W/2, H/2 - 2*cm, width=W/2 - margin*1.5, height=H/2, preserveAspectRatio=True, anchor='sw')
    except Exception:
        pass

    # Resumen de métricas
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Resumen de métricas (grados)")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    for k in ["neck_flexion_deg", "trunk_flexion_deg", "hip_right_deg", "knee_right_deg", "elbow_right_deg", "wrist_ext_right_deg"]:
        v = metrics.get(k)
        if v is not None:
            c.drawString(margin, y, f"• {k}: {v:.1f}")
            y -= 0.42*cm

    c.showPage()

    # ---------- Recomendaciones ----------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - margin, "Recomendaciones priorizadas")
    c.setFont("Helvetica", 10)
    y = H - margin - 1*cm

    if recommendations:
        for title, text in recommendations:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, f"• {title}")
            y -= 0.5*cm
            c.setFont("Helvetica", 10)
            for line in _wrap_text_lines(text, width=100):
                c.drawString(margin + 0.6*cm, y, line)
                y -= 0.42*cm
            y -= 0.3*cm
            if y < margin + 2*cm:
                c.showPage()
                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, H - margin, "Recomendaciones (continuación)")
                c.setFont("Helvetica", 10)
                y = H - margin - 1*cm
    else:
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, "No se detectan desviaciones relevantes respecto a los rangos recomendados.")

    c.showPage()

    # ---------- Anexo técnico ----------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - margin, "Anexo técnico y consideraciones")
    y = H - margin - 1*cm
    c.setFont("Helvetica", 10)
    lines = [
        "• Este informe se genera a partir de una única imagen (2D) y ofrece estimaciones de ángulos relevantes.",
        "• La precisión depende de la vista (ideal: lateral), iluminación, oclusiones y ropa.",
        "• Recomendaciones basadas en buenas prácticas en ergonomía de oficina (cuello <15°, tronco <20°, codo 90–100°, cadera 90–110°, rodilla ~90°, muñeca ~neutra).",
        "• No sustituye una evaluación completa in situ; úsese como cribado rápido para priorizar mejoras.",
    ]
    for line in lines:
        c.drawString(margin, y, line)
        y -= 0.5*cm

    # Colocar imagen original (opcional)
    y -= 0.3*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Imagen original")
    y -= 0.6*cm
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
            Image.fromarray(original_rgb).save(tmp2.name, format="PNG")
            c.drawImage(tmp2.name, margin, y - (H/3), width=W/2 - margin*1.5, height=H/3, preserveAspectRatio=True, anchor='sw')
    except Exception:
        pass

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()
