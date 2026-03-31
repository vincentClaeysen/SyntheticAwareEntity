#!/usr/bin/env python3
"""
Vision Primitive — OAK-D Lite / Webcam
ESPACE : analyser la frame courante et afficher l'overlay
ESPACE (à nouveau) : retour au flux temps réel
"""

import threading
import queue
from datetime import datetime

import cv2
import numpy as np
try:
    import depthai as dai
    DEPTHAI_DISPONIBLE = True
except ImportError:
    DEPTHAI_DISPONIBLE = False

# ─────────────────────────────────────────────────────────────
#  DÉTECTION PLATEFORME
# ─────────────────────────────────────────────────────────────
def _detecter_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False

IS_PI = _detecter_pi()

try:
    _test = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    WLS_DISPONIBLE = True
except AttributeError:
    WLS_DISPONIBLE = False

USE_WLS = WLS_DISPONIBLE and not IS_PI

def _detecter_oakd() -> bool:
    if not DEPTHAI_DISPONIBLE:
        return False
    try:
        return len(dai.Device.getAllAvailableDevices()) > 0
    except Exception:
        return False

USE_OAKD = _detecter_oakd()
print(f"[CAPTEUR] {'OAK-D Lite' if USE_OAKD else 'Webcam'}")


# ─────────────────────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────────────────────
W, H          = 640, 400 if USE_OAKD else 640, 480
FPS_ACQ       = 20
QUEUE_MAX     = 2
PRECISION     = 0.6
AF_PERIODE_S  = 2.0
AF_ZONE_FRAC  = 0.25
LENSPOS_MIN   = 0
LENSPOS_MAX   = 255


# ─────────────────────────────────────────────────────────────
#  COULEURS PAR TYPE D'OBJET
# ─────────────────────────────────────────────────────────────
COULEURS_PAR_LABEL = {
    "triangle":   (0, 165, 255),
    "carré":      (255, 0, 0),
    "rectangle":  (255, 255, 0),
    "trapèze":    (255, 165, 0),
    "pentagone":  (128, 0, 128),
    "hexagone":   (128, 64, 0),
    "heptagone":  (200, 100, 0),
    "octogone":   (200, 150, 0),
    "patatoïde":  (255, 0, 255),
    "cercle":     (0, 255, 0),
    "ovale":      (0, 200, 200),
    "polygone":   (0, 255, 255),
    "default":    (200, 200, 200)
}


# ─────────────────────────────────────────────────────────────
#  AUTOFOCUS (OAK-D)
# ─────────────────────────────────────────────────────────────
def distance_vers_lenspos(dist_mm: float) -> int:
    if dist_mm <= 0:
        return 120
    lp = int(1500 / dist_mm * 30)
    return int(np.clip(lp, LENSPOS_MIN, LENSPOS_MAX))

def distance_mediane_centrale(depth_map: np.ndarray) -> float:
    h, w = depth_map.shape
    dy = int(h * AF_ZONE_FRAC / 2)
    dx = int(w * AF_ZONE_FRAC / 2)
    cy, cx = h // 2, w // 2
    zone = depth_map[cy - dy:cy + dy, cx - dx:cx + dx]
    valides = zone[(zone > 100) & (zone < 15000)]
    return float(np.median(valides)) if valides.size > 0 else 0.0


# ─────────────────────────────────────────────────────────────
#  PIPELINE OAK-D
# ─────────────────────────────────────────────────────────────
def creer_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(W, H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(FPS_ACQ)
    cam_rgb.initialControl.setManualFocus(120)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setFps(FPS_ACQ)
    mono_r.setFps(FPS_ACQ)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(W, H)
    stereo.setSubpixel(True)

    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 60
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.temporalFilter.alpha = 0.4
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 200
    config.postProcessing.thresholdFilter.maxRange = 8000
    stereo.initialConfig.set(config)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    ctrl_in = pipeline.create(dai.node.XLinkIn)
    ctrl_in.setStreamName("control")
    ctrl_in.out.link(cam_rgb.inputControl)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_disp = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    xout_disp.setStreamName("disparity")
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    stereo.disparity.link(xout_disp.input)

    return pipeline


# ─────────────────────────────────────────────────────────────
#  FILTRAGE DEPTH
# ─────────────────────────────────────────────────────────────
DISP_LEVELS = 96
_wls_filter = None
if USE_WLS:
    _wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    _wls_filter.setLambda(8000)
    _wls_filter.setSigmaColor(1.5)

def affiner_depth(depth_raw, disp_raw, frame_gray):
    if USE_WLS and _wls_filter is not None:
        disp_f = disp_raw.astype(np.float32)
        filtered_disp = _wls_filter.filter(disp_f, frame_gray, disparity_map_right=None)
        filtered_disp = np.clip(filtered_disp, 0, DISP_LEVELS).astype(np.float32)
        mask_both = (filtered_disp > 0) & (disp_raw > 0)
        depth_out = depth_raw.astype(np.float32).copy()
        ratio = np.where(mask_both, disp_raw.astype(np.float32) / np.maximum(filtered_disp, 1), 1.0)
        depth_out = np.where(mask_both, depth_out * ratio, depth_out)
        return np.clip(depth_out, 0, 65535).astype(np.uint16)
    else:
        d_max = depth_raw.max() if depth_raw.max() > 0 else 1
        d_norm = (depth_raw.astype(np.float32) / d_max * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(d_norm, d=5, sigmaColor=50, sigmaSpace=50)
        return (filtered.astype(np.float32) / 255 * d_max).astype(np.uint16)


# ─────────────────────────────────────────────────────────────
#  THREAD ACQUISITION OAK-D
# ─────────────────────────────────────────────────────────────
def thread_acquisition_oak(device, frame_queue, stop_event):
    q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
    q_disp = device.getOutputQueue("disparity", maxSize=1, blocking=False) if USE_WLS else None
    q_ctrl = device.getInputQueue("control")

    dernier_af = datetime.now()
    lenspos_actuel = 120
    lenspos_cible = 120
    EMA_ALPHA = 0.25 if IS_PI else 0.35
    depth_ema = None

    while not stop_event.is_set():
        in_rgb = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        in_disp = q_disp.tryGet() if q_disp is not None else True

        if in_rgb is None or in_depth is None or in_disp is None:
            continue

        frame_bgr = in_rgb.getCvFrame()
        depth_raw = in_depth.getFrame()
        disp_raw = in_disp.getFrame().astype(np.uint8) if USE_WLS else None
        ts = datetime.now()

        lenspos_reel = in_rgb.getLensPosition()
        if lenspos_reel != -1 and abs(lenspos_reel - lenspos_cible) > 2:
            continue

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        depth_filtre = affiner_depth(depth_raw, disp_raw if disp_raw is not None else np.zeros_like(depth_raw, dtype=np.uint8), frame_gray)

        if depth_ema is None:
            depth_ema = depth_filtre.astype(np.float32)
        else:
            mask = depth_filtre > 0
            depth_ema[mask] = (EMA_ALPHA * depth_filtre[mask].astype(np.float32) + (1 - EMA_ALPHA) * depth_ema[mask])
        depth_final = np.clip(depth_ema, 0, 65535).astype(np.uint16)

        if (ts - dernier_af).total_seconds() >= AF_PERIODE_S:
            dist_mm = distance_mediane_centrale(depth_final)
            nouveau_lp = distance_vers_lenspos(dist_mm)
            if abs(nouveau_lp - lenspos_actuel) > 5:
                ctrl = dai.CameraControl()
                ctrl.setManualFocus(nouveau_lp)
                q_ctrl.send(ctrl)
                lenspos_actuel = nouveau_lp
                lenspos_cible = nouveau_lp
                depth_ema = None
            dernier_af = ts

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame_bgr, depth_final, ts))


# ─────────────────────────────────────────────────────────────
#  THREAD ACQUISITION WEBCAM
# ─────────────────────────────────────────────────────────────
def thread_acquisition_webcam(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS_ACQ)

    if not cap.isOpened():
        print("[ERREUR] Webcam inaccessible")
        stop_event.set()
        return

    print(f"[WEBCAM] {W}x{H} @ {FPS_ACQ}fps")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put((frame, None, datetime.now()))

    cap.release()


# ─────────────────────────────────────────────────────────────
#  FONCTIONS VISION
# ─────────────────────────────────────────────────────────────
def est_trapeze(sommets):
    if len(sommets) != 4:
        return False
    vecteurs = []
    for i in range(4):
        x1, y1 = sommets[i]
        x2, y2 = sommets[(i + 1) % 4]
        v = (x2 - x1, y2 - y1)
        norm = np.sqrt(v[0]**2 + v[1]**2)
        vecteurs.append((v[0]/norm, v[1]/norm) if norm > 0 else (0, 0))
    p1 = abs(vecteurs[0][0]*vecteurs[2][0] + vecteurs[0][1]*vecteurs[2][1]) > 0.92
    p2 = abs(vecteurs[1][0]*vecteurs[3][0] + vecteurs[1][1]*vecteurs[3][1]) > 0.92
    return p1 or p2


def tester_patatoide(cnt):
    if len(cnt) < 5:
        return False, None
    ellipse = cv2.fitEllipse(cnt)
    (ex, ey), (ea, eb), angle = ellipse
    if ea <= 0 or eb <= 0:
        return False, None
    excentricite = eb / ea
    aire_ellipse = np.pi * (ea / 2) * (eb / 2)
    ratio = cv2.contourArea(cnt) / aire_ellipse if aire_ellipse > 0 else 0
    if not (ratio > 0.60 and excentricite > 0.30):
        return False, None
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rx, ry = ea / 2, eb / 2
    pts = cnt[:, 0, :].astype(np.float32)
    dx = (pts[:, 0] - ex) * cos_a + (pts[:, 1] - ey) * sin_a
    dy = -(pts[:, 0] - ex) * sin_a + (pts[:, 1] - ey) * cos_a
    dist_norm = np.sqrt((dx / rx)**2 + (dy / ry)**2)
    return (True, ellipse) if float(np.std(dist_norm)) > 0.15 else (False, None)


def ellipse_en_points(ellipse, n=24):
    (ex, ey), (ea, eb), angle = ellipse
    rx, ry = ea / 2, eb / 2
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    pts = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        px, py = rx * np.cos(a), ry * np.sin(a)
        pts.append((int(ex + px*cos_a - py*sin_a), int(ey + px*sin_a + py*cos_a)))
    return pts


def simplifier_convexe(hull, n_max=8):
    perim = cv2.arcLength(hull, True)
    lo, hi, best = 0.005, 0.40, cv2.approxPolyDP(hull, 0.005 * perim, True)
    for _ in range(22):
        mid = (lo + hi) / 2
        approx = cv2.approxPolyDP(hull, mid * perim, True)
        if len(approx) <= n_max:
            best, hi = approx, mid
        else:
            lo = mid
    return best


def classifier_forme(sommets, label_pata=False):
    if label_pata:
        return "patatoïde"
    n = len(sommets)
    if n == 3:
        return "triangle"
    if n == 4:
        x, y, w, h = cv2.boundingRect(np.array(sommets))
        ratio = max(w, h) / (min(w, h) + 0.01)
        if est_trapeze(sommets):
            return "trapèze"
        return "carré" if ratio < 1.15 else "rectangle"
    return {5: "pentagone", 6: "hexagone", 7: "heptagone", 8: "octogone"}.get(n, "polygone")


def analyser_photometrie(img, masque=None):
    if masque is not None:
        v = img[masque > 0]
        return (float(np.mean(v)), float(np.std(v))) if v.size > 0 else (0., 0.)
    return float(np.mean(img)), float(np.std(img))


SEUIL_AIRE_RELATIVE = 0.05
PROFONDEUR_MAX = 2


def _extraire_formes(gray, aire_parente, precision, profondeur, lum_globale):
    h, w = gray.shape
    seuil_aire = max(aire_parente * SEUIL_AIRE_RELATIVE, 200)

    div = max(2, min(10, int(12 - precision * 8)))
    canny_low = int(80 - precision * 60)
    canny_high = int(150 - precision * 90)

    small = cv2.resize(gray, (max(1, w//div), max(1, h//div)), interpolation=cv2.INTER_AREA)
    morphed = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    recon = cv2.resize(morphed, (w, h), interpolation=cv2.INTER_LINEAR)
    edges = cv2.Canny(recon, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    objets = []
    masque_utilise = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        if cv2.contourArea(cnt) < seuil_aire:
            continue

        mask_test = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_test, [cnt], -1, 255, -1)
        if np.count_nonzero(cv2.bitwise_and(masque_utilise, mask_test)) > 0.5 * np.count_nonzero(mask_test):
            continue
        cv2.add(masque_utilise, mask_test, masque_utilise)

        pata, ellipse = tester_patatoide(cnt)
        if pata:
            sommets = ellipse_en_points(ellipse, n=24)
            label = "patatoïde"
        else:
            hull = cv2.convexHull(cnt)
            approx = simplifier_convexe(hull, n_max=8)
            sommets = [tuple(pt[0]) for pt in approx]
            label = classifier_forme(sommets)

        if len(sommets) < 3:
            continue

        mask_obj = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask_obj, [np.array(sommets, dtype=np.int32)], 255)
        lum_loc, std_loc = analyser_photometrie(gray, mask_obj)

        x, y, wb, hb = cv2.boundingRect(np.array(sommets))
        objets.append({
            "label": label,
            "profondeur": profondeur,
            "points": sommets,
            "bbox": (int(x), int(y), int(wb), int(hb)),
            "aire": int(cv2.contourArea(cnt)),
            "couleur": COULEURS_PAR_LABEL.get(label, COULEURS_PAR_LABEL["default"]),
            "enfants": []
        })

    return objets


def vision_primitive_complete(frame, depth_map=None, z_cible=None, precision=0.5, profondeur=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lum_globale, _ = analyser_photometrie(gray)

    if profondeur == 0 and depth_map is not None and z_cible is not None:
        z_min, z_max = max(0, z_cible - 250), z_cible + 250
        gray = np.where((depth_map >= z_min) & (depth_map <= z_max), gray, 0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    h, w = frame.shape[:2]
    aire_scene = float(w * h)

    objets = _extraire_formes(gray, aire_scene, precision, profondeur, lum_globale)

    if profondeur < PROFONDEUR_MAX:
        for obj in objets:
            bx, by, bw, bh = obj["bbox"]
            marge = int(min(bw, bh) * 0.10)
            x1 = max(0, bx - marge)
            y1 = max(0, by - marge)
            x2 = min(w, bx + bw + marge)
            y2 = min(h, by + bh + marge)

            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                continue

            aire_roi = float((x2 - x1) * (y2 - y1))
            if aire_roi < aire_scene * SEUIL_AIRE_RELATIVE:
                continue

            enfants = _extraire_formes(roi_gray, aire_roi, precision, profondeur + 1, lum_globale)

            for enfant in enfants:
                enfant["points"] = [(px + x1, py + y1) for px, py in enfant["points"]]
                ex, ey, ew, eh = enfant["bbox"]
                enfant["bbox"] = (ex + x1, ey + y1, ew, eh)

            obj["enfants"] = enfants

    return objets, {"lum_globale": round(lum_globale, 2)}


# ─────────────────────────────────────────────────────────────
#  THREAD TRAITEMENT
# ─────────────────────────────────────────────────────────────
def thread_traitement(frame_queue, analyse_queue, stop_event):
    derniere_frame = None
    derniere_depth = None

    while not stop_event.is_set():
        try:
            frame, depth, _ = frame_queue.get(timeout=0.05)
            derniere_frame = frame
            derniere_depth = depth
        except queue.Empty:
            pass

        try:
            analyse_queue.get_nowait()
            if derniere_frame is not None:
                z_cible = distance_mediane_centrale(derniere_depth) if derniere_depth is not None else 0.0
                objets, scene = vision_primitive_complete(
                    derniere_frame,
                    depth_map=derniere_depth,
                    z_cible=z_cible if z_cible > 0 else None,
                    precision=PRECISION
                )
                analyse_queue.put((derniere_frame.copy(), objets, scene))
            else:
                analyse_queue.put((None, [], {}))
        except queue.Empty:
            pass


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=QUEUE_MAX)
    analyse_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    if USE_OAKD:
        pipeline = creer_pipeline()
        device = dai.Device(pipeline)
        t_acq = threading.Thread(target=thread_acquisition_oak, args=(device, frame_queue, stop_event), daemon=True)
    else:
        t_acq = threading.Thread(target=thread_acquisition_webcam, args=(frame_queue, stop_event), daemon=True)

    t_trt = threading.Thread(target=thread_traitement, args=(frame_queue, analyse_queue, stop_event), daemon=True)

    t_acq.start()
    t_trt.start()

    mode_analyse = False
    frame_fige = None
    objets_figes = None
    scene_stats = None

    print("\n=== Vision Primitive ===")
    print("ESPACE : analyser la frame courante")
    print("q : quitter\n")

    while not stop_event.is_set():
        if mode_analyse and frame_fige is not None:
            # Affichage figé avec overlay
            aff = frame_fige.copy()
            for obj in objets_figes:
                pts = np.array(obj["points"], dtype=np.int32)
                couleur = obj["couleur"]
                cv2.drawContours(aff, [pts], -1, couleur, 2)
                x, y, w, h = obj["bbox"]
                cv2.rectangle(aff, (x, y), (x+w, y+h), couleur, 1)
                cv2.putText(aff, obj["label"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 1)
                for enfant in obj.get("enfants", []):
                    pts_e = np.array(enfant["points"], dtype=np.int32)
                    cv2.drawContours(aff, [pts_e], -1, enfant["couleur"], 1)
                    ex, ey, ew, eh = enfant["bbox"]
                    cv2.rectangle(aff, (ex, ey), (ex+ew, ey+eh), enfant["couleur"], 1)
            cv2.putText(aff, "ANALYSE FIGEE - ESPACE pour reprendre", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Vision Primitive", aff)
        else:
            try:
                frame, _, _ = frame_queue.get(timeout=0.05)
                cv2.imshow("Vision Primitive", frame)
            except queue.Empty:
                pass

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            stop_event.set()
        elif key == ord(' '):
            if not mode_analyse:
                print("📸 Capture...")
                try:
                    analyse_queue.get_nowait()
                except:
                    pass
                analyse_queue.put("analyse")
                try:
                    frame_fige, objets_figes, scene_stats = analyse_queue.get(timeout=2.0)
                    if frame_fige is not None:
                        mode_analyse = True
                        print(f"✅ {len(objets_figes)} objet(s) détecté(s)")
                    else:
                        print("❌ Pas de frame")
                except queue.Empty:
                    print("❌ Timeout")
            else:
                mode_analyse = False
                frame_fige = None
                print("🔄 Retour temps réel")

    if USE_OAKD:
        device.close()
    cv2.destroyAllWindows()
    print("Arrêt.")