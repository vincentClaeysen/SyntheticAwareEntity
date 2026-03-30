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
import depthai as dai

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
#  L'OAK-D Lite utilise une lentille motorisée (0–255).
#  On mappe la distance médiane (mm) → lens position.
#  Relation empirique : lensPos ≈ clip(1500/dist_mm * 30, 0, 255)
# ─────────────────────────────────────────────────────────────
def distance_vers_lenspos(dist_mm: float) -> int:
    if dist_mm <= 0:
        return 120  # valeur par défaut ~1 m
    lp = int(1500 / dist_mm * 30)
    return int(np.clip(lp, LENSPOS_MIN, LENSPOS_MAX))


def distance_mediane_centrale(depth_map: np.ndarray) -> float:
    """Distance médiane dans la zone centrale (AF_ZONE_FRAC x AF_ZONE_FRAC)."""
    h, w  = depth_map.shape
    dy    = int(h * AF_ZONE_FRAC / 2)
    dx    = int(w * AF_ZONE_FRAC / 2)
    cy, cx = h // 2, w // 2
    zone  = depth_map[cy - dy:cy + dy, cx - dx:cx + dx]
    valides = zone[(zone > 100) & (zone < 15000)]   # filtre valeurs aberrantes
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
    # Autofocus DÉSACTIVÉ — contrôle manuel de la lentille
    cam_rgb.initialControl.setManualFocus(120)

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
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # alignée sur RGB
    stereo.setOutputSize(W, H)
    # Sous-pixel actif → disparité plus fine, réduit l'effet "marches d'escalier"
    stereo.setSubpixel(True)

    # — Filtres post-stéréo (exécutés sur MyriadX, coût CPU hôte = 0) —
    config = stereo.initialConfig.get()

    # Speckle : supprime les îlots isolés de pixels aberrants
    config.postProcessing.speckleFilter.enable           = True
    config.postProcessing.speckleFilter.speckleRange     = 60

    # Temporal : lisse dans le temps (alpha=0.4 → réactivité correcte)
    config.postProcessing.temporalFilter.enable          = True
    config.postProcessing.temporalFilter.alpha           = 0.4

    # Spatial : lisse spatialement en respectant les bords (hole-filling inclus)
    config.postProcessing.spatialFilter.enable           = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations    = 1

    # Threshold : rejette les valeurs hors plage utile (évite les artefacts lointains)
    config.postProcessing.thresholdFilter.minRange       = 200    # mm
    config.postProcessing.thresholdFilter.maxRange       = 8000   # mm

    stereo.initialConfig.set(config)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    # — Contrôle caméra (pour mise au point manuelle dynamique) —
    ctrl_in = pipeline.create(dai.node.XLinkIn)
    ctrl_in.setStreamName("control")
    ctrl_in.out.link(cam_rgb.inputControl)

    # — Sorties vers l'hôte —
    xout_rgb   = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    # Disparity nécessaire pour le filtre WLS côté hôte
    xout_disp = pipeline.create(dai.node.XLinkOut)
    xout_disp.setStreamName("disparity")
    stereo.disparity.link(xout_disp.input)

    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return pipeline


# ─────────────────────────────────────────────────────────────
#  THREAD 1 — ACQUISITION
# ─────────────────────────────────────────────────────────────
DISP_LEVELS = 96

# Filtre WLS instancié une seule fois si disponible
_wls_filter = None
if USE_WLS:
    _wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    _wls_filter.setLambda(8000)
    _wls_filter.setSigmaColor(1.5)


def affiner_depth_wls(depth_raw: np.ndarray,
                      disp_raw: np.ndarray,
                      frame_gray: np.ndarray) -> np.ndarray:
    """
    Filtre WLS si disponible (PC), sinon filtre bilatéral léger (Pi).
    Les deux préservent les bords tout en lissant les zones homogènes.
    """
    if USE_WLS and _wls_filter is not None:
        # — Mode PC : WLS complet —
        disp_f        = disp_raw.astype(np.float32)
        filtered_disp = _wls_filter.filter(disp_f, frame_gray, disparity_map_right=None)
        filtered_disp = np.clip(filtered_disp, 0, DISP_LEVELS).astype(np.float32)
        mask_both     = (filtered_disp > 0) & (disp_raw > 0)
        depth_out     = depth_raw.astype(np.float32).copy()
        ratio         = np.where(mask_both,
                                 disp_raw.astype(np.float32) / np.maximum(filtered_disp, 1),
                                 1.0)
        depth_out     = np.where(mask_both, depth_out * ratio, depth_out)
        return np.clip(depth_out, 0, 65535).astype(np.uint16)
    else:
        # — Mode Pi : filtre bilatéral sur depth normalisée —
        # Convertit en uint8 pour bilateralFilter, re-scale après
        d_max    = depth_raw.max() if depth_raw.max() > 0 else 1
        d_norm   = (depth_raw.astype(np.float32) / d_max * 255).astype(np.uint8)
        # d=5 : voisinage petit → rapide sur Pi ; sigmaColor/sigmaSpace calibrés
        filtered = cv2.bilateralFilter(d_norm, d=5, sigmaColor=50, sigmaSpace=50)
        return (filtered.astype(np.float32) / 255 * d_max).astype(np.uint16)


def thread_acquisition(device: dai.Device,
                       frame_queue: queue.Queue,
                       stop_event: threading.Event):
    """
    Lit RGB + depth + disparity à 20 fps.
    Applique WLS + moyenne temporelle sur la depth.
    Pousse (frame_bgr, depth_lissée, datetime) dans frame_queue.
    Recalibre la lentille toutes les AF_PERIODE_S secondes.
    """
    q_rgb   = device.getOutputQueue("rgb",   maxSize=1, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
    q_disp  = device.getOutputQueue("disparity", maxSize=1, blocking=False) if USE_WLS else None
    q_ctrl  = device.getInputQueue("control")

    dernier_af     = datetime.now()
    lenspos_actuel = 120

    # EMA : alpha réduit sur Pi (lissage plus fort, moins de bruit résiduel)
    EMA_ALPHA = 0.25 if IS_PI else 0.35
    depth_ema = None

    while not stop_event.is_set():
        in_rgb   = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        in_disp  = q_disp.tryGet() if q_disp is not None else True   # True = pas bloquant

        if in_rgb is None or in_depth is None or in_disp is None:
            continue

        frame_bgr = in_rgb.getCvFrame()
        depth_raw = in_depth.getFrame()                                        # uint16, mm
        disp_raw  = in_disp.getFrame().astype(np.uint8) if USE_WLS else None  # uint8 ou None
        ts        = datetime.now()

        # — Filtrage depth (WLS sur PC, bilatéral sur Pi) —
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        depth_wls  = affiner_depth_wls(
            depth_raw,
            disp_raw if disp_raw is not None else np.zeros_like(depth_raw, dtype=np.uint8),
            frame_gray
        )

        # — Moyenne temporelle EMA —
        if depth_ema is None:
            depth_ema = depth_wls.astype(np.float32)
        else:
            # Pixels invalides (0) ne contribuent pas à l'EMA
            mask = depth_wls > 0
            depth_ema[mask] = (EMA_ALPHA * depth_wls[mask].astype(np.float32)
                               + (1 - EMA_ALPHA) * depth_ema[mask])
        depth_final = np.clip(depth_ema, 0, 65535).astype(np.uint16)

        # — Recalibration autofocus périodique —
        delta = (ts - dernier_af).total_seconds()
        if delta >= AF_PERIODE_S:
            dist_mm    = distance_mediane_centrale(depth_final)
            nouveau_lp = distance_vers_lenspos(dist_mm)
            if abs(nouveau_lp - lenspos_actuel) > 5:
                ctrl = dai.CameraControl()
                ctrl.setManualFocus(nouveau_lp)
                q_ctrl.send(ctrl)
                lenspos_actuel = nouveau_lp
            dernier_af = ts

        # — Push dans la queue —
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame_bgr, depth_final, ts))


# ─────────────────────────────────────────────────────────────
#  THREAD 2 — TRAITEMENT (vision_primitive_complete)
# ─────────────────────────────────────────────────────────────

def est_trapeze(sommets):
    """Détecte si un quadrilatère est un trapèze (au moins une paire de côtés parallèles)"""
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
    """
    Détecte les formes imparfaites (coins arrondis, déformations)
    Retourne (True, ellipse) si c'est une forme patatoïde
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
    pts = cnt[:, 0, :].astype(np.float32)
    dx =  (pts[:, 0] - ex) * cos_a + (pts[:, 1] - ey) * sin_a
    dy = -(pts[:, 0] - ex) * sin_a + (pts[:, 1] - ey) * cos_a
    dist_norm = np.sqrt((dx / rx)**2 + (dy / ry)**2)
    return (True, ellipse) if float(np.std(dist_norm)) > 0.15 else (False, None)


def ellipse_en_points(ellipse, n=24):
    """Convertit une ellipse en liste de points"""
    (ex, ey), (ea, eb), angle = ellipse
    rx, ry = ea / 2, eb / 2
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    pts = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        px, py = rx * np.cos(a), ry * np.sin(a)
        pts.append((int(ex + px*cos_a - py*sin_a), int(ey + px*sin_a + py*cos_a)))
    return pts


def simplifier_convexe(hull, n_max=8):
    """Simplifie un contour convexe en limitant le nombre de sommets"""
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
    """
    Classification bio-inspirée des formes
    - 3: triangle
    - 4: carré / rectangle / trapèze
    - 5: pentagone
    - 6: hexagone
    - 7+: arrondi (cercle, ovale) ou patatoïde
    """
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
    
    if n == 5:
        return "pentagone"
    
    if n == 6:
        return "hexagone"
    
    # n >= 7 : forme arrondie
    # On distingue cercle vs ovale par l'ellipticité
    try:
        pts = np.array(sommets, dtype=np.int32)
        ellipse = cv2.fitEllipse(pts)
        axes = ellipse[1]
        if min(axes) > 0:
            ellipticite = min(axes) / max(axes)
            if ellipticite > 0.85:
                return "cercle"
            else:
                return "ovale"
    except:
        pass
    
    return "arrondi"


def analyser_photometrie(image_gris, masque=None):
    """Analyse la luminosité et le contraste d'une région"""
    if masque is not None:
        v = image_gris[masque > 0]
        return (float(np.mean(v)), float(np.std(v))) if v.size > 0 else (0., 0.)
    return float(np.mean(image_gris)), float(np.std(image_gris))


def vision_primitive_complete(frame, depth_map=None, z_cible=None, precision=0.5):
    """
    Pipeline de vision bio-inspirée
    Retourne: (recon, edges, objets, scene_stats)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    lum_globale, std_globale = analyser_photometrie(gray)

    # Masquage par profondeur (optionnel)
    if depth_map is not None and z_cible is not None:
        z_min, z_max = max(0, z_cible - 250), z_cible + 250
        gray = np.where((depth_map >= z_min) & (depth_map <= z_max), gray, 0).astype(np.uint8)

    # CLAHE pour améliorer le contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Downsampling adaptatif selon précision
    div = max(2, min(10, int(12 - precision * 8)))
    canny_low = int(80 - precision * 60)
    canny_high = int(150 - precision * 90)

    small = cv2.resize(gray, (max(1, w//div), max(1, h//div)), interpolation=cv2.INTER_AREA)
    
    # Morphologie : ouverture puis fermeture elliptique
    morphed = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    recon = cv2.resize(morphed, (w, h), interpolation=cv2.INTER_LINEAR)
    edges = cv2.Canny(recon, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    seuil_aire = (w * h) / 300

    objets = []
    masque_utilise = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        if cv2.contourArea(cnt) < seuil_aire:
            continue
        
        # Éviter les redondances (détections qui se chevauchent)
        mask_test = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_test, [cnt], -1, 255, -1)
        if np.count_nonzero(cv2.bitwise_and(masque_utilise, mask_test)) > \
           0.5 * np.count_nonzero(mask_test):
            continue
        cv2.add(masque_utilise, mask_test, masque_utilise)

        # Test patatoïde (forme imparfaite)
        pata, ellipse = tester_patatoide(cnt)
        
        if pata:
            sommets = ellipse_en_points(ellipse, n=24)
            label = classifier_forme(sommets, label_pata=True)
            couleur = (255, 0, 255)  # magenta pour patatoïde
        else:
            hull = cv2.convexHull(cnt)
            approx = simplifier_convexe(hull, n_max=8)
            sommets = [tuple(pt[0]) for pt in approx]
            label = classifier_forme(sommets, label_pata=False)
            couleur = (0, 255, 255)  # jaune pour formes régulières

        if len(sommets) < 3:
            continue

        # Photométrie locale
        mask_obj = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask_obj, [np.array(sommets, dtype=np.int32)], 255)
        lum_loc, std_loc = analyser_photometrie(gray, mask_obj)

        x, y, wb, hb = cv2.boundingRect(np.array(sommets))
        objets.append({
            "label": label,
            "n_sommets": len(sommets),
            "points": sommets,
            "centre": (int(x + wb/2), int(y + hb/2)),
            "aire": int(cv2.contourArea(cnt)),
            "couleur": couleur,
            "photometrie": {
                "lum_locale": round(lum_loc, 2),
                "std_locale": round(std_loc, 2),
                "contraste_relatif": round(lum_loc / (lum_globale + 0.1), 2)
            }
        })

    scene_stats = {
        "lum_globale": round(lum_globale, 2),
        "std_globale": round(std_globale, 2)
    }

    return recon, edges, objets, scene_stats


def thread_traitement(frame_queue: queue.Queue,
                      stop_event: threading.Event):
    """
    Dépile la queue, exécute la vision, affiche le résultat.
    La z_cible est la distance médiane centrale de la depth map reçue.
    """
    while not stop_event.is_set():
        try:
            frame_bgr, depth_map, ts = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Distance cible = médiane centrale de la depth map
        z_cible = distance_mediane_centrale(depth_map)

        _, edges, objets, scene = vision_primitive_complete(
            frame_bgr,
            depth_map=depth_map,
            z_cible=z_cible if z_cible > 0 else None,
            precision=PRECISION
        )

        # — Affichage —
        affichage = frame_bgr.copy()
        for obj in objets:
            pts = np.array(obj["points"], dtype=np.int32)
            couleur = obj["couleur"]
            cv2.drawContours(affichage, [pts], -1, couleur, 2)
            info = f"{obj['label']} std:{obj['photometrie']['std_locale']:.0f}"
            cv2.putText(affichage, info, obj["points"][0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, couleur, 1)

        latence_ms = (datetime.now() - ts).total_seconds() * 1000
        hud = (f"lum:{scene['lum_globale']:.0f}  "
               f"std:{scene['std_globale']:.0f}  "
               f"z:{z_cible:.0f}mm  "
               f"lat:{latence_ms:.0f}ms  "
               f"{ts.strftime('%H:%M:%S.%f')[:-3]}")
        cv2.putText(affichage, hud, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Depth colorisée en incrustation (coin bas-droit)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_col = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        dh, dw = H // 4, W // 4
        depth_small = cv2.resize(depth_col, (dw, dh))
        affichage[H - dh:H, W - dw:W] = depth_small

        cv2.imshow("Vision Primitive — OAK-D Lite", affichage)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = creer_pipeline()
    frame_queue = queue.Queue(maxsize=QUEUE_MAX)
    stop_event = threading.Event()

    with dai.Device(pipeline) as device:
        t_acq = threading.Thread(
            target=thread_acquisition,
            args=(device, frame_queue, stop_event),
            daemon=True,
            name="Acquisition"
        )
        t_trt = threading.Thread(
            target=thread_traitement,
            args=(frame_queue, stop_event),
            daemon=True,
            name="Traitement"
        )

        t_acq.start()
        t_trt.start()

        print("Vision Primitive OAK-D Lite — Q pour quitter")
        stop_event.wait()   # attend signal d'arrêt (touche Q dans le thread traitement)

        t_acq.join(timeout=2)
        t_trt.join(timeout=2)

    cv2.destroyAllWindows()
    print("Arrêt propre.")