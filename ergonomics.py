# ergonomics.py
import math
import numpy as np
import cv2
from ultralytics import YOLO

# Mapa COCO keypoints (YOLOv8-pose)
KP = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# Estructura de esqueleto para dibujar
SKELETON = [
    (KP['right_ankle'], KP['right_knee']),
    (KP['right_knee'], KP['right_hip']),
    (KP['right_hip'], KP['right_shoulder']),
    (KP['right_shoulder'], KP['right_elbow']),
    (KP['right_elbow'], KP['right_wrist']),
    (KP['left_ankle'], KP['left_knee']),
    (KP['left_knee'], KP['left_hip']),
    (KP['left_hip'], KP['left_shoulder']),
    (KP['left_shoulder'], KP['left_elbow']),
    (KP['left_elbow'], KP['left_wrist']),
    (KP['left_shoulder'], KP['right_shoulder']),
    (KP['left_hip'], KP['right_hip']),
]

_model = None

def _get_model():
    global _model
    if _model is None:
        # Descarga automática de pesos en el primer uso (yolov8n-pose)
        _model = YOLO('yolov8n-pose.pt')
    return _model

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


def _get_side_points(kp_xy, side='right'):
    # Devuelve dict con puntos clave de un lado si existen
    idx = KP
    s = 'right' if side == 'right' else 'left'
    pts = {
        'ear': kp_xy[idx[f'{s}_ear']] if kp_xy[idx[f'{s}_ear']] is not None else None,
        'shoulder': kp_xy[idx[f'{s}_shoulder']] if kp_xy[idx[f'{s}_shoulder']] is not None else None,
        'elbow': kp_xy[idx[f'{s}_elbow']] if kp_xy[idx[f'{s}_elbow']] is not None else None,
        'wrist': kp_xy[idx[f'{s}_wrist']] if kp_xy[idx[f'{s}_wrist']] is not None else None,
        'hip': kp_xy[idx[f'{s}_hip']] if kp_xy[idx[f'{s}_hip']] is not None else None,
        'knee': kp_xy[idx[f'{s}_knee']] if kp_xy[idx[f'{s}_knee']] is not None else None,
        'ankle': kp_xy[idx[f'{s}_ankle']] if kp_xy[idx[f'{s}_ankle']] is not None else None,
    }
    return pts


def _choose_person(results):
    # Elige la persona con mayor confianza media en keypoints o mayor conf de caja
    if len(results) == 0:
        return None
    r = results[0]
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        return None
    kps = r.keypoints
    # Seleccionar primer individuo (o el de mayor conf si hay varios)
    idx_person = 0
    if kps.conf is not None:
        conf_mean = kps.conf.mean(axis=1)  # (n_personas,)
        idx_person = int(np.argmax(conf_mean))
    # Extraer xy de esa persona
    xy = kps.xy[idx_person].cpu().numpy()  # (17, 2)
    conf = None
    if kps.conf is not None:
        conf = kps.conf[idx_person].cpu().numpy()
    return xy, conf


def _as_xy_list(xy_arr, conf_arr=None, min_conf=0.2):
    # Convierte arrays a lista de (x,y) o None si confianza baja
    out = []
    for i in range(xy_arr.shape[0]):
        p = tuple(map(float, xy_arr[i]))
        if conf_arr is not None:
            if float(conf_arr[i]) < min_conf:
                out.append(None)
                continue
        out.append(p)
    return out


def compute_metrics_from_kp(kp_xy):
    # Intenta lado derecho; si faltan puntos clave, usa izquierdo
    right = _get_side_points(kp_xy, 'right')
    left = _get_side_points(kp_xy, 'left')

    side = right
    side_name = 'right'
    # Si faltan hombro/codo/muñeca derechas, probar izquierda
    needed = ['shoulder', 'hip', 'knee']
    if any(side[k] is None for k in needed):
        side = left
        side_name = 'left'

    metrics = {}

    # Cuello: oreja-hombro vs vertical
    if side['ear'] is not None and side['shoulder'] is not None:
        metrics['neck_flexion_deg'] = angle_from_vertical(side['ear'], side['shoulder'], side['ear'])

    # Tronco y cadera: hombro-cadera-rodilla
    if side['shoulder'] is not None and side['hip'] is not None and side['knee'] is not None:
        metrics['trunk_flexion_deg'] = angle_3pts(side['shoulder'], side['hip'], side['knee'])
        metrics[f'hip_{side_name}_deg'] = metrics['trunk_flexion_deg']

    # Rodilla: cadera-rodilla-tobillo
    if side['hip'] is not None and side['knee'] is not None and side['ankle'] is not None:
        metrics[f'knee_{side_name}_deg'] = angle_3pts(side['hip'], side['knee'], side['ankle'])

    # Codo: hombro-codo-muñeca
    if side['shoulder'] is not None and side['elbow'] is not None and side['wrist'] is not None:
        metrics[f'elbow_{side_name}_deg'] = angle_3pts(side['shoulder'], side['elbow'], side['wrist'])

    # Muñeca (proxy): ángulo del antebrazo respecto a horizontal
    if side['elbow'] is not None and side['wrist'] is not None:
        v = np.array([side['wrist'][0] - side['elbow'][0], side['wrist'][1] - side['elbow'][1]])
        horizontal = np.array([1, 0])
        denom = (np.linalg.norm(v) * np.linalg.norm(horizontal) + 1e-9)
        cosang = np.dot(v, horizontal) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        forearm_ang = np.degrees(np.arccos(cosang))
        metrics[f'wrist_ext_{side_name}_deg'] = float(forearm_ang)

    # Normaliza claves para que app.py muestre right_* si existe, si no left_*
    def _prefer_right(key_base):
        k_r = f"{key_base}_right_deg"
        k_l = f"{key_base}_left_deg"
        if k_r in metrics:
            metrics[key_base + '_right_deg'] = metrics[k_r]
        elif k_l in metrics:
            # duplica a clave 'right' para que UI lo muestre
            metrics[key_base + '_right_deg'] = metrics[k_l]

    for base in ['hip', 'knee', 'elbow', 'wrist_ext']:
        _prefer_right(base)

    return metrics


def analyze_posture(img_bgr):
    model = _get_model()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(source=img_rgb, verbose=False)
    sel = _choose_person(res)
    if sel is None:
        return {"metrics": None, "recommendations": None, "landmarks_xy": None, "skeleton_edges": SKELETON}
    xy, conf = sel
    kp_xy = _as_xy_list(xy, conf_arr=conf, min_conf=0.15)

    metrics = compute_metrics_from_kp(kp_xy)
    recs = build_recommendations(metrics)
    return {"metrics": metrics, "recommendations": recs, "landmarks_xy": kp_xy, "skeleton_edges": SKELETON}


def draw_landmarks_bgr(img_bgr, kp_xy, skeleton_edges=None):
    out = img_bgr.copy()
    if kp_xy is None:
        return out

    # Dibujar esqueleto
    if skeleton_edges is None:
        skeleton_edges = SKELETON
    for a, b in skeleton_edges:
        if kp_xy[a] is not None and kp_xy[b] is not None:
            pa = tuple(int(v) for v in kp_xy[a])
            pb = tuple(int(v) for v in kp_xy[b])
            cv2.line(out, pa, pb, (0, 200, 255), 2)

    # Dibujar puntos
    for p in kp_xy:
        if p is None:
            continue
        cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
    return out

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
