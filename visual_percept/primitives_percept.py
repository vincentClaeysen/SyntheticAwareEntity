#!/usr/bin/env python3
"""
Vision Primitive — OAK-D Lite
Thread 1 : acquisition RGB + depth alignée à 20 fps → Queue
Thread 2 : traitement vision_primitive_complete + affichage
Autofocus : désactivé, recalibré périodiquement sur la distance médiane
            de la zone centrale de la depth map.
"""

import os
import platform
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
#  Sur Raspberry Pi, ximgproc peut être absent ou trop lent.
#  Le WLS est désactivé automatiquement dans ce cas.
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

USE_WLS = WLS_DISPONIBLE and not IS_PI   # désactivé sur Pi (trop coûteux)

if IS_PI:
    print("[PLATEFORME] Raspberry Pi détecté — WLS désactivé, mode allégé actif")
else:
    print(f"[PLATEFORME] PC/autre — WLS={'ON' if USE_WLS else 'OFF (ximgproc absent)'}")

def _detecter_oakd() -> bool:
    """Tente de trouver un device OAK-D connecté."""
    if not DEPTHAI_DISPONIBLE:
        return False
    try:
        return len(dai.Device.getAllAvailableDevices()) > 0
    except Exception:
        return False

USE_OAKD = _detecter_oakd()
print(f"[CAPTEUR] {'OAK-D Lite détecté' if USE_OAKD else 'Pas d OAK-D — fallback webcam'}")


# ─────────────────────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────────────────────
W, H          = 640, 400          # Résolution native OAK-D Lite (ratio 16/10)
FPS_ACQ       = 20                # Fréquence d'acquisition
QUEUE_MAX     = 2                 # On ne conserve que les frames fraîches
PRECISION     = 0.6               # Paramètre pipeline vision
AF_PERIODE_S  = 2.0               # Recalibration autofocus toutes les N secondes
AF_ZONE_FRAC  = 0.25              # Zone centrale depth pour calcul distance (fraction)
LENSPOS_MIN   = 0                 # Plage lentille OAK-D Lite
LENSPOS_MAX   = 255


# ─────────────────────────────────────────────────────────────
#  AUTOFOCUS → LENS POSITION
# ─────────────────────────────────────────────────────────────
def distance_vers_lenspos(dist_mm: float) -> int:
    if dist_mm <= 0:
        return 120
    lp = int(1500 / dist_mm * 30)
    return int(np.clip(lp, LENSPOS_MIN, LENSPOS_MAX))


def distance_mediane_centrale(depth_map: np.ndarray) -> float:
    """Distance médiane dans la zone centrale (AF_ZONE_FRAC x AF_ZONE_FRAC)."""
    h, w   = depth_map.shape
    dy     = int(h * AF_ZONE_FRAC / 2)
    dx     = int(w * AF_ZONE_FRAC / 2)
    cy, cx = h // 2, w // 2
    zone   = depth_map[cy - dy:cy + dy, cx - dx:cx + dx]
    valides = zone[(zone > 100) & (zone < 15000)]
    return float(np.median(valides)) if valides.size > 0 else 0.0


# ─────────────────────────────────────────────────────────────
#  PIPELINE DEPTHAI
# ─────────────────────────────────────────────────────────────
def creer_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # — Caméra RGB —
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(W, H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(FPS_ACQ)
    cam_rgb.initialControl.setManualFocus(120)   # autofocus désactivé

    # — Caméras stéréo (mono L + R) —
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setFps(FPS_ACQ)
    mono_r.setFps(FPS_ACQ)

    # — Depth —
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(W, H)
    stereo.setSubpixel(True)   # disparité plus fine, réduit l'effet marches d'escalier

    # Filtres post-stéréo sur MyriadX (coût CPU hôte = 0)
    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable            = True
    config.postProcessing.speckleFilter.speckleRange      = 60
    config.postProcessing.temporalFilter.enable           = True
    config.postProcessing.temporalFilter.alpha            = 0.4
    config.postProcessing.spatialFilter.enable            = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations     = 1
    config.postProcessing.thresholdFilter.minRange        = 200   # mm
    config.postProcessing.thresholdFilter.maxRange        = 8000  # mm
    stereo.initialConfig.set(config)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    # — Contrôle caméra —
    ctrl_in = pipeline.create(dai.node.XLinkIn)
    ctrl_in.setStreamName("control")
    ctrl_in.out.link(cam_rgb.inputControl)

    # — Sorties vers l'hôte —
    xout_rgb   = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_disp  = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    xout_disp.setStreamName("disparity")
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    stereo.disparity.link(xout_disp.input)   # nécessaire pour WLS

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


def affiner_depth(depth_raw: np.ndarray,
                  disp_raw:  np.ndarray,
                  frame_gray: np.ndarray) -> np.ndarray:
    """
    PC  → WLS : lisse en respectant les bords RGB (meilleure qualité).
    Pi  → filtre bilatéral : préserve les bords, rapide sur ARM.
    """
    if USE_WLS and _wls_filter is not None:
        disp_f        = disp_raw.astype(np.float32)
        filtered_disp = _wls_filter.filter(disp_f, frame_gray, disparity_map_right=None)
        filtered_disp = np.clip(filtered_disp, 0, DISP_LEVELS).astype(np.float32)
        mask_both     = (filtered_disp > 0) & (disp_raw > 0)
        depth_out     = depth_raw.astype(np.float32).copy()
        ratio         = np.where(mask_both,
                                 disp_raw.astype(np.float32) / np.maximum(filtered_disp, 1),
                                 1.0)
        depth_out = np.where(mask_both, depth_out * ratio, depth_out)
        return np.clip(depth_out, 0, 65535).astype(np.uint16)
    else:
        d_max  = depth_raw.max() if depth_raw.max() > 0 else 1
        d_norm = (depth_raw.astype(np.float32) / d_max * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(d_norm, d=5, sigmaColor=50, sigmaSpace=50)
        return (filtered.astype(np.float32) / 255 * d_max).astype(np.uint16)


# ─────────────────────────────────────────────────────────────
#  THREAD 1 — ACQUISITION
# ─────────────────────────────────────────────────────────────
def thread_acquisition(device: dai.Device,
                       frame_queue: queue.Queue,
                       stop_event: threading.Event):
    """
    Lit RGB + depth (+ disparity si WLS) à 20 fps.
    Applique filtrage depth + EMA temporelle.
    Recalibre la lentille périodiquement.
    Pousse (frame_bgr, depth_finale, datetime) dans frame_queue.
    """
    q_rgb   = device.getOutputQueue("rgb",   maxSize=1, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
    q_disp  = device.getOutputQueue("disparity", maxSize=1, blocking=False) if USE_WLS else None
    q_ctrl  = device.getInputQueue("control")

    dernier_af     = datetime.now()
    lenspos_actuel = 120
    lenspos_cible  = 120   # preset envoyé — on attend que la frame le confirme

    EMA_ALPHA = 0.25 if IS_PI else 0.35   # plus lissé sur Pi
    depth_ema = None

    while not stop_event.is_set():
        in_rgb   = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        in_disp  = q_disp.tryGet() if q_disp is not None else True   # True = pas bloquant

        if in_rgb is None or in_depth is None or in_disp is None:
            continue

        frame_bgr   = in_rgb.getCvFrame()
        depth_raw   = in_depth.getFrame()                                        # uint16, mm
        disp_raw    = in_disp.getFrame().astype(np.uint8) if USE_WLS else None
        ts          = datetime.now()

        # Vérification focus hardware : on ignore la frame si la lentille
        # n'a pas encore atteint la position cible.
        # getLensPosition() est une métadonnée du capteur RGB, indépendante de l'IMU.
        # Tolérance ±2 : oscillation normale de la lentille motorisée.
        # Retourne -1 si non disponible → on laisse passer.
        lenspos_reel = in_rgb.getLensPosition()
        if lenspos_reel != -1 and abs(lenspos_reel - lenspos_cible) > 2:
            continue

        # Filtrage depth
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        depth_filtre = affiner_depth(
            depth_raw,
            disp_raw if disp_raw is not None else np.zeros_like(depth_raw, dtype=np.uint8),
            frame_gray
        )

        # EMA temporelle (pixels invalides exclus)
        if depth_ema is None:
            depth_ema = depth_filtre.astype(np.float32)
        else:
            mask = depth_filtre > 0
            depth_ema[mask] = (EMA_ALPHA * depth_filtre[mask].astype(np.float32)
                               + (1 - EMA_ALPHA) * depth_ema[mask])
        depth_final = np.clip(depth_ema, 0, 65535).astype(np.uint16)

        # Recalibration autofocus périodique
        if (ts - dernier_af).total_seconds() >= AF_PERIODE_S:
            dist_mm    = distance_mediane_centrale(depth_final)
            nouveau_lp = distance_vers_lenspos(dist_mm)
            if abs(nouveau_lp - lenspos_actuel) > 5:
                ctrl = dai.CameraControl()
                ctrl.setManualFocus(nouveau_lp)
                q_ctrl.send(ctrl)
                lenspos_actuel = nouveau_lp
                lenspos_cible  = nouveau_lp   # le filtre hardware prend le relais
                depth_ema      = None          # reset EMA : frames floues exclues
            dernier_af = ts
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame_bgr, depth_final, ts))


# ─────────────────────────────────────────────────────────────
#  THREAD 2 — VISION
# ─────────────────────────────────────────────────────────────

def est_trapeze(sommets):
    """Détecte si un quadrilatère a au moins une paire de côtés parallèles."""
    if len(sommets) != 4:
        return False
    vecteurs = []
    for i in range(4):
        x1, y1 = sommets[i]
        x2, y2 = sommets[(i + 1) % 4]
        v    = (x2 - x1, y2 - y1)
        norm = np.sqrt(v[0]**2 + v[1]**2)
        vecteurs.append((v[0]/norm, v[1]/norm) if norm > 0 else (0, 0))
    p1 = abs(vecteurs[0][0]*vecteurs[2][0] + vecteurs[0][1]*vecteurs[2][1]) > 0.92
    p2 = abs(vecteurs[1][0]*vecteurs[3][0] + vecteurs[1][1]*vecteurs[3][1]) > 0.92
    return p1 or p2


def tester_patatoide(cnt):
    """
    Patatoïde = ellipse organique irrégulière.
    Condition 1 : le contour s'inscrit globalement dans une ellipse (ratio > 0.60,
                  excentricité > 0.30).
    Condition 2 : il dévie suffisamment de l'ellipse parfaite (écart-type des
                  distances normalisées > 0.15) — sinon c'est un ovale régulier.
    """
    if len(cnt) < 5:
        return False, None
    ellipse = cv2.fitEllipse(cnt)
    (ex, ey), (ea, eb), angle = ellipse
    if ea <= 0 or eb <= 0:
        return False, None
    excentricite = eb / ea
    aire_ellipse = np.pi * (ea / 2) * (eb / 2)
    ratio        = cv2.contourArea(cnt) / aire_ellipse if aire_ellipse > 0 else 0
    if not (ratio > 0.60 and excentricite > 0.30):
        return False, None
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rx, ry = ea / 2, eb / 2
    pts    = cnt[:, 0, :].astype(np.float32)
    dx     =  (pts[:, 0] - ex) * cos_a + (pts[:, 1] - ey) * sin_a
    dy     = -(pts[:, 0] - ex) * sin_a + (pts[:, 1] - ey) * cos_a
    dist_norm = np.sqrt((dx / rx)**2 + (dy / ry)**2)
    return (True, ellipse) if float(np.std(dist_norm)) > 0.15 else (False, None)


def ellipse_en_points(ellipse, n=24):
    """Discrétise une ellipse fitEllipse en N sommets (représentation finale)."""
    (ex, ey), (ea, eb), angle = ellipse
    rx, ry = ea / 2, eb / 2
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    pts = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        px, py = rx * np.cos(a), ry * np.sin(a)
        pts.append((int(ex + px*cos_a - py*sin_a), int(ey + px*sin_a + py*cos_a)))
    return pts


def simplifier_convexe(hull, n_max=8):
    """Polygone convexe ≤ n_max côtés par recherche binaire sur epsilon."""
    perim = cv2.arcLength(hull, True)
    lo, hi, best = 0.005, 0.40, cv2.approxPolyDP(hull, 0.005 * perim, True)
    for _ in range(22):
        mid    = (lo + hi) / 2
        approx = cv2.approxPolyDP(hull, mid * perim, True)
        if len(approx) <= n_max:
            best, hi = approx, mid
        else:
            lo = mid
    return best


def classifier_forme(sommets, label_pata=False):
    """
    Classification géométrique des formes.
    Les patatoïdes sont déjà identifiés en amont — cette fonction
    ne reçoit que des polygones convexes ≤ 8 côtés.
    """
    if label_pata:
        return "patatoïde"
    n = len(sommets)
    if n == 3: return "triangle"
    if n == 4:
        x, y, w, h = cv2.boundingRect(np.array(sommets))
        ratio = max(w, h) / (min(w, h) + 0.01)
        if est_trapeze(sommets): return "trapèze"
        return "carré" if ratio < 1.15 else "rectangle"
    # n == 5..8 : nommage direct, pas d'ambiguïté possible
    # (les formes rondes ont été captées par tester_patatoide avant)
    return {5: "pentagone", 6: "hexagone", 7: "heptagone", 8: "octogone"}.get(n, f"poly_{n}")


def analyser_photometrie(image_gris, masque=None):
    """Luminance moyenne + écart-type (homogénéité)."""
    if masque is not None:
        v = image_gris[masque > 0]
        return (float(np.mean(v)), float(np.std(v))) if v.size > 0 else (0., 0.)
    return float(np.mean(image_gris)), float(np.std(image_gris))


# Seuil d'arrêt récursion : sous-forme < 5% de l'aire parente
SEUIL_AIRE_RELATIVE = 0.05
# Profondeur max : 2 niveaux (scène → formes → sous-formes)
PROFONDEUR_MAX = 2

# Couleurs par niveau de profondeur
COULEURS_NIVEAU = {
    0: (0, 255, 255),    # cyan   — niveau scène
    1: (255, 165, 0),    # orange — sous-formes niveau 1
    2: (0, 255, 0),      # vert   — sous-formes niveau 2
}


def _extraire_formes(gray: np.ndarray,
                     aire_parente: float,
                     precision: float,
                     profondeur: int,
                     lum_globale: float) -> list:
    """
    Noyau partagé d'extraction de formes sur une image grise (ROI ou scène).
    Retourne une liste d'objets sans récursion (appelée par vision_primitive_complete).
    Le seuil d'aire minimum est relatif à l'aire du parent.
    """
    h, w = gray.shape
    seuil_aire = max(aire_parente * SEUIL_AIRE_RELATIVE, 200)

    div        = max(2, min(10, int(12 - precision * 8)))
    canny_low  = int(80  - precision * 60)
    canny_high = int(150 - precision * 90)

    small   = cv2.resize(gray, (max(1, w//div), max(1, h//div)), interpolation=cv2.INTER_AREA)
    morphed = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    recon   = cv2.resize(morphed, (w, h), interpolation=cv2.INTER_LINEAR)
    edges   = cv2.Canny(recon, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    objets         = []
    masque_utilise = np.zeros((h, w), dtype=np.uint8)
    couleur_niv    = COULEURS_NIVEAU.get(profondeur, (128, 128, 128))

    for cnt in contours:
        if cv2.contourArea(cnt) < seuil_aire:
            continue

        mask_test = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_test, [cnt], -1, 255, -1)
        if np.count_nonzero(cv2.bitwise_and(masque_utilise, mask_test)) >            0.5 * np.count_nonzero(mask_test):
            continue
        cv2.add(masque_utilise, mask_test, masque_utilise)

        pata, ellipse = tester_patatoide(cnt)
        if pata:
            sommets = ellipse_en_points(ellipse, n=24)
            label   = "patatoïde"
            couleur = (255, 0, 255) if profondeur == 0 else couleur_niv
        else:
            hull    = cv2.convexHull(cnt)
            approx  = simplifier_convexe(hull, n_max=8)
            sommets = [tuple(pt[0]) for pt in approx]
            label   = classifier_forme(sommets)
            couleur = couleur_niv

        if len(sommets) < 3:
            continue

        mask_obj = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask_obj, [np.array(sommets, dtype=np.int32)], 255)
        lum_loc, std_loc = analyser_photometrie(gray, mask_obj)

        aire_cnt = int(cv2.contourArea(cnt))
        x, y, wb, hb = cv2.boundingRect(np.array(sommets))
        objets.append({
            "label"      : label,
            "profondeur" : profondeur,
            "n_sommets"  : len(sommets),
            "points"     : sommets,
            "centre"     : (int(x + wb/2), int(y + hb/2)),
            "bbox"       : (int(x), int(y), int(wb), int(hb)),
            "aire"       : aire_cnt,
            "couleur"    : couleur,
            "photometrie": {
                "lum_locale"       : round(lum_loc, 2),
                "std_locale"       : round(std_loc, 2),
                "contraste_relatif": round(lum_loc / (lum_globale + 0.1), 2)
            },
            "enfants"    : []   # rempli par la récursion
        })

    return objets


def vision_primitive_complete(frame, depth_map=None, z_cible=None, precision=0.5,
                               profondeur=0):
    """
    Pipeline de vision bio-inspirée récursif.
    - profondeur=0 : analyse de la scène complète
    - profondeur=1 : analyse des ROI de chaque forme détectée
    Retourne : (recon, edges, objets, scene_stats)
    Chaque objet contient un champ "enfants" avec ses sous-formes.
    """
    h, w  = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    lum_globale, std_globale = analyser_photometrie(gray)

    # Z-cut (bulle de perception) — niveau scène uniquement
    if profondeur == 0 and depth_map is not None and z_cible is not None:
        z_min, z_max = max(0, z_cible - 250), z_cible + 250
        gray = np.where((depth_map >= z_min) & (depth_map <= z_max), gray, 0).astype(np.uint8)

    # CLAHE — révèle les contours à faible contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    aire_scene = float(w * h)

    # Reconstruction pour affichage (retournée uniquement au niveau 0)
    div     = max(2, min(10, int(12 - precision * 8)))
    small   = cv2.resize(gray, (max(1, w//div), max(1, h//div)), interpolation=cv2.INTER_AREA)
    morphed = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    recon   = cv2.resize(morphed, (w, h), interpolation=cv2.INTER_LINEAR)
    canny_low  = int(80  - precision * 60)
    canny_high = int(150 - precision * 90)
    edges   = cv2.Canny(recon, canny_low, canny_high)

    # Extraction des formes au niveau courant
    objets = _extraire_formes(gray, aire_scene, precision, profondeur, lum_globale)

    # ── RÉCURSION (niveau suivant si pas encore au max) ──────────────────────
    if profondeur < PROFONDEUR_MAX:
        for obj in objets:
            bx, by, bw, bh = obj["bbox"]

            # Marge de 10% autour de la bbox pour ne pas couper les bords
            marge  = int(min(bw, bh) * 0.10)
            x1     = max(0, bx - marge)
            y1     = max(0, by - marge)
            x2     = min(w,  bx + bw + marge)
            y2     = min(h,  by + bh + marge)

            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                continue

            aire_roi = float((x2 - x1) * (y2 - y1))

            # Seuil : la ROI doit être suffisamment grande pour contenir
            # des sous-formes significatives (> SEUIL_AIRE_RELATIVE de la scène)
            if aire_roi < aire_scene * SEUIL_AIRE_RELATIVE:
                continue

            enfants = _extraire_formes(
                roi_gray,
                aire_roi,          # aire parente = aire de la ROI
                precision,
                profondeur + 1,
                lum_globale
            )

            # Recalage des coordonnées enfants dans le repère de la scène
            for enfant in enfants:
                enfant["points"] = [
                    (px + x1, py + y1) for px, py in enfant["points"]
                ]
                enfant["centre"] = (
                    enfant["centre"][0] + x1,
                    enfant["centre"][1] + y1
                )
                ex, ey, ew, eh = enfant["bbox"]
                enfant["bbox"] = (ex + x1, ey + y1, ew, eh)

            obj["enfants"] = enfants

    return recon, edges, objets, {
        "lum_globale": round(lum_globale, 2),
        "std_globale": round(std_globale, 2)
    }


def thread_traitement(frame_queue: queue.Queue,
                      stop_event: threading.Event):
    """
    Dépile la queue, exécute la vision, affiche le résultat.
    """
    while not stop_event.is_set():
        try:
            frame_bgr, depth_map, ts = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        z_cible = distance_mediane_centrale(depth_map) if depth_map is not None else 0.0

        _, edges, objets, scene = vision_primitive_complete(
            frame_bgr,
            depth_map=depth_map,
            z_cible=z_cible if z_cible > 0 else None,
            precision=PRECISION
        )

        affichage = frame_bgr.copy()

        def dessiner_objet(canvas, obj):
            """Dessine récursivement objet + ses enfants."""
            pts     = np.array(obj["points"], dtype=np.int32)
            couleur = obj["couleur"]
            epaisseur = 2 if obj["profondeur"] == 0 else 1
            cv2.drawContours(canvas, [pts], -1, couleur, epaisseur)
            # Label : profondeur indentée visuellement par la taille de fonte
            fonte = max(0.3, 0.45 - obj["profondeur"] * 0.1)
            info  = f"{'  ' * obj['profondeur']}{obj['label']}"
            cv2.putText(canvas, info, obj["points"][0],
                        cv2.FONT_HERSHEY_SIMPLEX, fonte, couleur, 1)
            for enfant in obj.get("enfants", []):
                dessiner_objet(canvas, enfant)

        for obj in objets:
            dessiner_objet(affichage, obj)

        latence_ms = (datetime.now() - ts).total_seconds() * 1000
        z_str = f"z:{z_cible:.0f}mm" if depth_map is not None else "z:N/A"
        hud = (f"lum:{scene['lum_globale']:.0f}  "
               f"std:{scene['std_globale']:.0f}  "
               f"{z_str}  "
               f"lat:{latence_ms:.0f}ms  "
               f"{ts.strftime('%H:%M:%S.%f')[:-3]}")
        cv2.putText(affichage, hud, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Depth colorisée en incrustation (coin bas-droit) — OAK-D seulement
        if depth_map is not None:
            depth_vis   = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_col   = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            dh, dw      = H // 4, W // 4
            depth_small = cv2.resize(depth_col, (dw, dh))
            affichage[H - dh:H, W - dw:W] = depth_small

        titre = "Vision Primitive — OAK-D Lite" if USE_OAKD else "Vision Primitive — Webcam"
        cv2.imshow(titre, affichage)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()


# ─────────────────────────────────────────────────────────────
#  THREAD 1b — ACQUISITION WEBCAM (fallback sans OAK-D)
#  Pas de depth map → depth_map=None, z_cible désactivé.
#  Pas d'autofocus géré (webcam fixe).
# ─────────────────────────────────────────────────────────────
def thread_acquisition_webcam(frame_queue: queue.Queue,
                               stop_event: threading.Event,
                               index: int = 0):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS,          FPS_ACQ)

    if not cap.isOpened():
        print(f"[ERREUR] Webcam index {index} inaccessible.")
        stop_event.set()
        return

    print(f"[WEBCAM] Ouverture index {index} — {W}x{H} @ {FPS_ACQ}fps")

    while not stop_event.is_set():
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        # Resize si la webcam n'a pas respecté la résolution demandée
        fh, fw = frame_bgr.shape[:2]
        if fw != W or fh != H:
            frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)

        ts = datetime.now()

        # Pas de depth map en mode webcam
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame_bgr, None, ts))

    cap.release()


# ─────────────────────────────────────────────────────────────
#  THREAD TRAITEMENT — adapté depth optionnelle
# ─────────────────────────────────────────────────────────────
# (thread_traitement gère déjà depth_map=None via vision_primitive_complete)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=QUEUE_MAX)
    stop_event  = threading.Event()

    t_trt = threading.Thread(
        target=thread_traitement,
        args=(frame_queue, stop_event),
        daemon=True, name="Traitement"
    )

    if USE_OAKD:
        pipeline = creer_pipeline()
        with dai.Device(pipeline) as device:
            t_acq = threading.Thread(
                target=thread_acquisition,
                args=(device, frame_queue, stop_event),
                daemon=True, name="Acquisition-OAK"
            )
            t_acq.start()
            t_trt.start()
            print("Vision Primitive OAK-D Lite — Q pour quitter")
            stop_event.wait()
            t_acq.join(timeout=2)
            t_trt.join(timeout=2)
    else:
        t_acq = threading.Thread(
            target=thread_acquisition_webcam,
            args=(frame_queue, stop_event),
            daemon=True, name="Acquisition-Webcam"
        )
        t_acq.start()
        t_trt.start()
        print("Vision Primitive Webcam — Q pour quitter")
        stop_event.wait()
        t_acq.join(timeout=2)
        t_trt.join(timeout=2)

    cv2.destroyAllWindows()
    print("Arrêt propre.")
