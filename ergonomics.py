# ergonomics.py
import math
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
POSE_LM = mp_pose.PoseLandmark

# ---------- Utilidades geométricas ----------
def angle_3pts(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def angle_from_vertical(p_top, p_bottom, p_ref):
    # Ángulo del vector (bottom->ref) respecto a vertical hacia arriba
    v = np.array([p_ref[0] - p_bottom[0], p_ref[1] - p_bottom[1]])
    vertical = np.array([0, -1])
    denom = np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-9
    cosang = np.dot(v, vertical) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _safe(pts, idx):
    if pts is None or idx is None or idx >= len(pts):
        return None
    return pts[idx]


# ---------- Reglas de recomendación ----------
def build_recommendations(metrics):
    recs = []

    neck = metrics.get("neck_flexion_deg")
    if neck is not None and neck > 15:
        recs.append((
            "Cuello flexionado",
            f"Reduce la flexión del cuello (≈ {neck:.0f}°). Eleva la pantalla hasta el nivel de los ojos "
            "y acerca la silla/teclado para evitar inclinar el cuello hacia delante."
        ))

    trunk = metrics.get("trunk_flexion_deg")
    if trunk is not None and trunk > 20:
        recs.append((
            "Tronco inclinado",
            f"Apoya la espalda en el respaldo (tronco ≈ {trunk:.0f}°). Ajusta profundidad del asiento y "
            "usa soporte lumbar; acerca la silla a la mesa."
        ))

    elbow = metrics.get("elbow_right_deg")
    if elbow is not None and (elbow < 80 or elbow > 110):
        recs.append((
            "Ángulo de codo fuera de rango",
            f"Ajusta altura de silla/brazos y acerca teclado/ratón (codo ≈ {elbow:.0f}°). Objetivo 90–100°."
        ))

    wrist = metrics.get("wrist_ext_right_deg")
    if wrist is not None and abs(wrist) > 15:
        recs.append((
            "Muñeca en desviación",
            f"Reduce extensión/flexión de la muñeca (≈ {wrist:.0f}°). Baja/aleja ligeramente el teclado "
            "y considera un reposamuñecas blando."
        ))

    hip = metrics.get("hip_right_deg")
    if hip is not None and (hip < 90 or hip > 120):
        recs.append((
            "Ángulo de cadera no óptimo",
            f"Ajusta altura de la silla (cadera ≈ {hip:.0f}°). Objetivo 90–110°."
        ))

    knee = metrics.get("knee_right_deg")
    if knee is not None and (knee < 80 or knee > 100):
        recs.append((
            "Ángulo de rodilla no óptimo",
            f"Ajusta la altura de la silla o usa reposapiés (rodilla ≈ {knee:.0f}°). Objetivo ~90°."
        ))

    priority = ["Cuello flexionado", "Tronco inclinado", "Ángulo de codo fuera de rango",
                "Muñeca en desviación", "Ángulo de cadera no óptimo", "Ángulo de rodilla no óptimo"]
    recs_sorted = sorted(recs, key=lambda x: priority.index(x[0]) if x[0] in priority else 99)
    return recs_sorted


# ---------- Pose + métricas ----------
def _pose_landmarks_bgr(img_bgr):
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None, None
        h, w = img_bgr.shape[:2]
        pts = [(lm.x * w, lm.y * h) for lm in res.pose_landmarks.landmark]
        return pts, res.pose_landmarks


def compute_metrics(pts):
    g = lambda lm: _safe(pts, lm.value if hasattr(lm, "value") else int(lm))

    ear_r = g(POSE_LM.RIGHT_EAR)
    shoulder_r = g(POSE_LM.RIGHT_SHOULDER)
    elbow_r = g(POSE_LM.RIGHT_ELBOW)
    wrist_r = g(POSE_LM.RIGHT_WRIST)
    hip_r = g(POSE_LM.RIGHT_HIP)
    knee_r = g(POSE_LM.RIGHT_KNEE)
    ankle_r = g(POSE_LM.RIGHT_ANKLE)

    metrics = {}

    # Cuello: aproximación con oreja-hombro vs vertical
    if ear_r and shoulder_r:
        metrics["neck_flexion_deg"] = angle_from_vertical(ear_r, shoulder_r, ear_r)

    # Tronco (proxy): hombro-cadera-rodilla
    if shoulder_r and hip_r and knee_r:
        metrics["trunk_flexion_deg"] = angle_3pts(shoulder_r, hip_r, knee_r)

    # Codo: hombro-codo-muñeca
    if shoulder_r and elbow_r and wrist_r:
        metrics["elbow_right_deg"] = angle_3pts(shoulder_r, elbow_r, wrist_r)

    # Muñeca (proxy): orientación del antebrazo respecto a horizontal
    if elbow_r and wrist_r:
        v = np.array([wrist_r[0] - elbow_r[0], wrist_r[1] - elbow_r[1]])
        horizontal = np.array([1, 0])
        denom = (np.linalg.norm(v) * np.linalg.norm(horizontal) + 1e-9)
        cosang = np.dot(v, horizontal) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        forearm_ang = np.degrees(np.arccos(cosang))
        metrics["wrist_ext_right_deg"] = forearm_ang

    # Cadera: hombro-cadera-rodilla
    if shoulder_r and hip_r and knee_r:
        metrics["hip_right_deg"] = angle_3pts(shoulder_r, hip_r, knee_r)

    # Rodilla: cadera-rodilla-tobillo
    if hip_r and knee_r and ankle_r:
        metrics["knee_right_deg"] = angle_3pts(hip_r, knee_r, ankle_r)

    return metrics


def draw_landmarks_bgr(img_bgr, pts):
    out = img_bgr.copy()
    if pts is None:
        return out
    for (x, y) in pts:
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out


def analyze_posture(img_bgr):
    pts, _ = _pose_landmarks_bgr(img_bgr)
    if pts is None:
        return {"metrics": None, "recommendations": None, "landmarks_xy": None}
    metrics = compute_metrics(pts)
    recs = build_recommendations(metrics)
    return {"metrics": metrics, "recommendations": recs, "landmarks_xy": pts}
