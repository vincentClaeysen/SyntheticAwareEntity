#!/usr/bin/env python3
"""
retina_lab.py — Rétine primitive + eFovea pour Shirka
=====================================================
Stade : jeune enfant (humeur ignorée)

Pipeline :
  1. Rétine périphérique basse résolution → carte de saillance brute
  2. Filtrage GEM (couleurs préférentielles + réactivité)
  3. Détection de multiples POIs avec pondération (saillance, mouvement, contraste, intersections)
  4. Sélection et classement des POIs
  5. eFovea : élargissement par depth map → bbox physique de l'objet
  6. Assainissement des ROIs (filtrage décrochage Z)
  7. Détection des plans (sol, plafond) et lignes structurelles
  8. Visualisation : chemin attentionnel + mires + heatmap optionnelle

Touche ESPACE : analyse de la frame courante
Touche H      : toggle heatmap
Touche Q      : quitter
"""

import math
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
#  GEM — Shirka_001
# ─────────────────────────────────────────────────────────────
GEM = {
    "identifiant" : "Shirka_001",
    "naissance"   : "2026-03-07T10:30:00Z",
    "genre"       : "omega",
    "tempo"       : {"base": 0.65, "variabilite": 0.3},
    "intensite"   : {"base": 0.7,  "seuil": 0.4},
    "grace"       : 0.5,
    "reactivite"  : 0.8,
    "emotive"     : 0.6,
    "courbe"      : "exponentielle",
    "couleur"     : {
        "principale" : "#3A6EA5",
        "secondaire" : "#E8C1A0",
        "intensite"  : 0.7
    },
    "resonance"   : {
        "frequence"   : 432,
        "largeur"     : 0.3,
        "harmoniques" : [864, 1296]
    },
    "preferences" : {"direction": 0.3, "interaction": 0.7, "esthetique": 0.5},
}


def hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


COULEUR_PRINCIPALE = hex_to_bgr(GEM["couleur"]["principale"])
COULEUR_SECONDAIRE = hex_to_bgr(GEM["couleur"]["secondaire"])
REACTIVITE         = GEM["reactivite"]
INTENSITE_SEUIL    = GEM["intensite"]["seuil"]


# ─────────────────────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────────────────────
W, H              = 640, 400
FPS_ACQ           = 20
QUEUE_MAX         = 2
DIV_RETINE        = 4
AF_ZONE_FRAC      = 0.25
LENSPOS_MIN       = 0
LENSPOS_MAX       = 255
AF_PERIODE_S      = 2.0
EFOVEA_MARGE_FRAC = 0.15


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
print(f"[GEM] {GEM['identifiant']} — genre:{GEM['genre']} — réactivité:{REACTIVITE}")


# ─────────────────────────────────────────────────────────────
#  UTILITAIRES DEPTH
# ─────────────────────────────────────────────────────────────
def distance_mediane_centrale(depth_map: np.ndarray) -> float:
    h, w    = depth_map.shape
    dy      = int(h * AF_ZONE_FRAC / 2)
    dx      = int(w * AF_ZONE_FRAC / 2)
    cy, cx  = h // 2, w // 2
    zone    = depth_map[cy-dy:cy+dy, cx-dx:cx+dx]
    valides = zone[(zone > 100) & (zone < 15000)]
    return float(np.median(valides)) if valides.size > 0 else 0.0


def distance_vers_lenspos(dist_mm: float) -> int:
    if dist_mm <= 0:
        return 120
    return int(np.clip(int(1500 / dist_mm * 30), LENSPOS_MIN, LENSPOS_MAX))


# ─────────────────────────────────────────────────────────────
#  CARTES DE SAILLANCE
# ─────────────────────────────────────────────────────────────
def carte_saillance_brute(frame_small: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY) \
            if len(frame_small.shape) == 3 else frame_small.copy()
    lap   = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    grad  = np.abs(lap)
    blur  = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
    ecart = np.abs(gray.astype(np.float32) - blur)
    carte = 0.5 * grad + 0.5 * ecart
    mn, mx = carte.min(), carte.max()
    if mx > mn:
        carte = (carte - mn) / (mx - mn)
    return carte.astype(np.float32)


def carte_saillance_couleur(frame_small: np.ndarray) -> np.ndarray:
    h, w  = frame_small.shape[:2]
    bonus = np.zeros((h, w), dtype=np.float32)
    frame_f = frame_small.astype(np.float32)
    for couleur_ref, poids in [
        (np.array(COULEUR_PRINCIPALE, dtype=np.float32), 1.0),
        (np.array(COULEUR_SECONDAIRE, dtype=np.float32), 0.6),
    ]:
        diff      = frame_f - couleur_ref
        dist      = np.sqrt(np.sum(diff**2, axis=2)) / (math.sqrt(3) * 255)
        proximite = np.clip(1.0 - dist, 0, 1)
        tolerance = 0.3 + GEM["grace"] * 0.4
        proximite = np.where(proximite > (1 - tolerance), proximite, 0)
        bonus    += proximite * poids
    bonus *= GEM["couleur"]["intensite"] * REACTIVITE
    return np.clip(bonus, 0, 1).astype(np.float32)


def carte_saillance_gem(frame_small: np.ndarray) -> np.ndarray:
    brute   = carte_saillance_brute(frame_small)
    couleur = carte_saillance_couleur(frame_small)
    carte   = brute + couleur * REACTIVITE
    mn, mx  = carte.min(), carte.max()
    if mx > mn:
        carte = (carte - mn) / (mx - mn)
    return carte.astype(np.float32)


def carte_mouvement(frame_curr: np.ndarray,
                    frame_prev: np.ndarray | None) -> np.ndarray:
    if frame_prev is None:
        return np.zeros(
            (frame_curr.shape[0] // DIV_RETINE,
             frame_curr.shape[1] // DIV_RETINE), dtype=np.float32
        )
    h, w   = frame_curr.shape[:2]
    sw, sh = max(1, w // DIV_RETINE), max(1, h // DIV_RETINE)
    curr_g = cv2.cvtColor(
        cv2.resize(frame_curr, (sw, sh), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_g = cv2.cvtColor(
        cv2.resize(frame_prev, (sw, sh), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff   = np.abs(curr_g - prev_g)
    mn, mx = diff.min(), diff.max()
    if mx > mn:
        diff = (diff - mn) / (mx - mn)
    return diff.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  STATISTIQUES SCÈNE
# ─────────────────────────────────────────────────────────────
def stats_scene(frame_small: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        "lum_moy": float(np.mean(hsv[:,:,2])),
        "lum_std": float(np.std(hsv[:,:,2])),
        "sat_moy": float(np.mean(hsv[:,:,1])),
    }


def carte_contraste_scene(frame_small: np.ndarray, stats: dict) -> np.ndarray:
    hsv       = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    ecart_lum = np.clip(
        np.abs(hsv[:,:,2] - stats["lum_moy"]) / (stats["lum_std"] + 1e-5) / 3.0,
        0, 1
    )
    ecart_sat = np.clip(hsv[:,:,1] / 255.0, 0, 1)
    carte     = 0.7 * ecart_lum + 0.3 * ecart_sat
    mn, mx    = carte.min(), carte.max()
    if mx > mn:
        carte = (carte - mn) / (mx - mn)
    return carte.astype(np.float32)


def carte_intersections(frame_small: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    corners = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)
    corners = np.clip(corners, 0, None)
    mn, mx  = corners.min(), corners.max()
    if mx > mn:
        corners = (corners - mn) / (mx - mn)
    return (corners * GEM["preferences"]["esthetique"]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  PROFONDEUR ET COHÉRENCE
# ─────────────────────────────────────────────────────────────
def z_ref_poi(cx: int, cy: int, depth_map: np.ndarray | None) -> float | None:
    if depth_map is None:
        return None
    z = float(depth_map[min(cy, depth_map.shape[0]-1),
                         min(cx, depth_map.shape[1]-1)])
    return z if 100 < z < 15000 else None


def coherence_z_poi(cx: int, cy: int, depth_map: np.ndarray | None,
                    rayon: int = 8) -> float:
    if depth_map is None:
        return 0.5
    h, w   = depth_map.shape
    zone   = depth_map[max(0,cy-rayon):min(h,cy+rayon),
                       max(0,cx-rayon):min(w,cx+rayon)].astype(np.float32)
    valide = zone[(zone > 100) & (zone < 15000)]
    if valide.size < 4:
        return 0.0
    cv = float(np.std(valide)) / (float(np.mean(valide)) + 1e-5)
    return float(max(0.0, 1.0 - cv * 5))


def poids_z(z_ref: float | None, coherence: float,
            depth_map: np.ndarray | None) -> float:
    if z_ref is None or depth_map is None:
        return 0.3
    prox = max(0.0, 1.0 - (z_ref - 200) / (8000 - 200))
    return float(prox * coherence)


# ─────────────────────────────────────────────────────────────
#  eFOVEA
# ─────────────────────────────────────────────────────────────
def efovea_bbox(poi: tuple, frame_h: int, frame_w: int,
                depth_map: np.ndarray | None) -> tuple:
    cx, cy = poi

    def bbox_fixe():
        bw = int(frame_w * 0.20)
        bh = int(frame_h * 0.20)
        return (max(0, cx - bw//2), max(0, cy - bh//2),
                min(frame_w, cx + bw//2), min(frame_h, cy + bh//2))

    if depth_map is None:
        return bbox_fixe()

    z_ref = float(depth_map[min(cy, depth_map.shape[0]-1),
                             min(cx, depth_map.shape[1]-1)])
    if z_ref < 100 or z_ref > 15000:
        return bbox_fixe()

    masque_z    = ((depth_map >= z_ref * 0.85) &
                   (depth_map <= z_ref * 1.15) &
                   (depth_map > 0)).astype(np.uint8) * 255
    masque_fill = masque_z.copy()
    seed        = (min(cx, masque_fill.shape[1]-1),
                   min(cy, masque_fill.shape[0]-1))
    mask_flood  = np.zeros((masque_fill.shape[0]+2,
                            masque_fill.shape[1]+2), dtype=np.uint8)
    cv2.floodFill(masque_fill, mask_flood, seed, 128,
                  loDiff=0, upDiff=0,
                  flags=cv2.FLOODFILL_MASK_ONLY | (128 << 8))
    region = (mask_flood[1:-1, 1:-1] == 128)

    if not region.any():
        return bbox_fixe()

    ys, xs  = np.where(region)
    marge_x = int((xs.max() - xs.min()) * EFOVEA_MARGE_FRAC)
    marge_y = int((ys.max() - ys.min()) * EFOVEA_MARGE_FRAC)
    return (max(0, xs.min() - marge_x), max(0, ys.min() - marge_y),
            min(frame_w, xs.max() + marge_x), min(frame_h, ys.max() + marge_y))


# ─────────────────────────────────────────────────────────────
#  ASSAINISSEMENT ROI
# ─────────────────────────────────────────────────────────────
def assainir_roi(roi_bgr: np.ndarray,
                 roi_depth: np.ndarray | None,
                 sigma_seuil: float = 2.0) -> tuple:
    """
    Supprime les pixels en décrochage Z (approche adaptative 2σ).
    Compatible webcam (roi_depth=None → masque plein, ROI inchangée).
    """
    if roi_depth is None:
        return roi_bgr.copy(), np.ones(roi_bgr.shape[:2], dtype=np.uint8)*255, None, None

    h, w = roi_bgr.shape[:2]
    if roi_depth.shape != (h, w):
        roi_depth = cv2.resize(roi_depth.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(np.uint16)

    depth_f = roi_depth.astype(np.float32)
    valides  = depth_f[(depth_f > 100) & (depth_f < 15000)]

    if valides.size < 10:
        return roi_bgr.copy(), np.ones((h, w), dtype=np.uint8)*255, None, None

    z_ref   = float(np.median(valides))
    sigma_z = float(np.std(valides))
    seuil_z = max(50.0, sigma_seuil * sigma_z)

    masque_bool  = ((depth_f > 100) & (depth_f < 15000) &
                    (np.abs(depth_f - z_ref) <= seuil_z))
    masque       = (masque_bool * 255).astype(np.uint8)
    roi_assainie = roi_bgr.copy()
    roi_assainie[~masque_bool] = 0

    return roi_assainie, masque, z_ref, sigma_z


# ─────────────────────────────────────────────────────────────
#  DÉTECTION DES PLANS ET STRUCTURES (vectorisée)
# ─────────────────────────────────────────────────────────────
def detecter_plans_principaux(depth_map: np.ndarray | None) -> dict:
    if depth_map is None:
        return {"sol": None, "plafond": None}
    valides = depth_map[(depth_map > 100) & (depth_map < 15000)]
    if valides.size == 0:
        return {"sol": None, "plafond": None}
    hist, bins = np.histogram(valides, bins=50)
    centres    = (bins[:-1] + bins[1:]) / 2
    # Pics locaux au-dessus de 30% du max
    seuil_pic = hist.max() * 0.3
    pics_idx  = np.where(
        (hist > seuil_pic) &
        (hist > np.roll(hist, 1)) &
        (hist > np.roll(hist, -1))
    )[0]
    pics = sorted(centres[pics_idx].tolist())
    return {
        "sol"    : float(pics[0])  if pics else None,
        "plafond": float(pics[-1]) if pics else None,
    }


def detecter_lignes_verticales(depth_map: np.ndarray | None,
                                sol: float | None,
                                plafond: float | None,
                                tolerance_z: float = 150) -> list:
    """Détection vectorisée des discontinuités verticales ancrées sol/plafond."""
    if depth_map is None or (sol is None and plafond is None):
        return []

    h, w = depth_map.shape
    depth_f = depth_map.astype(np.float32)

    # Gradient vertical par diff numpy (évite la double boucle)
    diff_y = np.abs(np.diff(depth_f, axis=0))   # (h-1, w)
    seuil  = np.percentile(diff_y[diff_y > 0], 95) if (diff_y > 0).any() else 100.0

    # Masques sol/plafond
    masque_sol    = np.zeros((h, w), dtype=bool)
    masque_plafond = np.zeros((h, w), dtype=bool)
    if sol      is not None:
        masque_sol     = np.abs(depth_f - sol)     < tolerance_z
    if plafond  is not None:
        masque_plafond = np.abs(depth_f - plafond) < tolerance_z

    lignes = []
    # Positions des discontinuités fortes
    ys, xs = np.where(diff_y > seuil)
    for y, x in zip(ys[::3], xs[::3]):   # sous-échantillonnage ×3 pour perf
        if masque_sol[y, x]:
            lignes.append({"type": "sol",     "position": (int(x), int(y)),
                           "z": float(depth_f[y, x])})
        elif masque_plafond[y, x]:
            lignes.append({"type": "plafond", "position": (int(x), int(y)),
                           "z": float(depth_f[y, x])})
    return lignes


def detecter_lignes_horizontales(depth_map: np.ndarray | None,
                                  sol: float | None,
                                  plafond: float | None,
                                  tolerance_z: float = 150,
                                  longueur_min: int = 30) -> list:
    """Détection vectorisée des lignes horizontales proches sol/plafond."""
    if depth_map is None or (sol is None and plafond is None):
        return []

    h, w    = depth_map.shape
    depth_f = depth_map.astype(np.float32)
    lignes  = []

    for ref_z, type_plan in [(sol, "sol"), (plafond, "plafond")]:
        if ref_z is None:
            continue
        # Masque lignes proches du plan
        masque = (np.abs(depth_f - ref_z) < tolerance_z) & \
                 (depth_f > 100) & (depth_f < 15000)
        # Par ligne horizontale, chercher les runs continus
        for y in range(0, h, 5):
            ligne_mask = masque[y, :]
            if not ligne_mask.any():
                continue
            # Détection des runs (début/fin de séquences True)
            changes = np.diff(ligne_mask.astype(np.int8))
            debuts  = np.where(changes ==  1)[0] + 1
            fins    = np.where(changes == -1)[0] + 1
            if ligne_mask[0]:
                debuts = np.concatenate([[0], debuts])
            if ligne_mask[-1]:
                fins = np.concatenate([fins, [w]])
            for d, f in zip(debuts, fins):
                if f - d >= longueur_min:
                    z_seg = float(np.median(depth_f[y, d:f]))
                    lignes.append({
                        "type"    : type_plan,
                        "position": (int(d), int(y)),
                        "largeur" : int(f - d),
                        "z"       : z_seg,
                    })
    return lignes


def detecter_intersections_3d(depth_map: np.ndarray | None,
                               sol: float | None,
                               plafond: float | None,
                               lignes_verticales: list,
                               lignes_horizontales: list,
                               dist_max: float = 30.0) -> list:
    """
    Intersections verticales + horizontales du même type (sol ou plafond).
    Utilise la distance euclidienne 2D plutôt qu'un simple écart X.
    """
    intersections = []
    for lv in lignes_verticales:
        xv, yv = lv["position"]
        for lh in lignes_horizontales:
            if lv["type"] != lh["type"]:
                continue
            xh, yh = lh["position"]
            dist = math.sqrt((xv - xh)**2 + (yv - yh)**2)
            if dist < dist_max:
                intersections.append({
                    "type"     : f"coin_{lv['type']}",
                    "position" : (int((xv+xh)//2), int((yv+yh)//2)),
                    "z"        : lv["z"],
                    "confiance": round(max(0.0, 1.0 - dist/dist_max), 2),
                })
    return intersections


# ─────────────────────────────────────────────────────────────
#  DÉTECTION MULTI-POIs
# ─────────────────────────────────────────────────────────────
def detecter_pois(frame: np.ndarray,
                  frame_prev: np.ndarray | None,
                  depth_map: np.ndarray | None,
                  n_max: int = 6) -> list:
    h, w   = frame.shape[:2]
    sw, sh = max(1, w // DIV_RETINE), max(1, h // DIV_RETINE)
    sx, sy = w / sw, h / sh

    frame_small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    scene_stats = stats_scene(frame_small)

    carte_sal   = carte_saillance_gem(frame_small)
    carte_mov   = carte_mouvement(frame, frame_prev)
    carte_ctr   = carte_contraste_scene(frame_small, scene_stats)
    carte_inter = carte_intersections(frame_small)

    if depth_map is not None:
        depth_small = cv2.resize(depth_map.astype(np.float32), (sw, sh),
                                 interpolation=cv2.INTER_AREA)
        valide = (depth_small > 100) & (depth_small < 15000)
        carte_prox = np.zeros((sh, sw), dtype=np.float32)
        if valide.any():
            d_min = depth_small[valide].min()
            d_max = depth_small[valide].max()
            if d_max > d_min:
                carte_prox[valide] = 1.0 - (depth_small[valide]-d_min)/(d_max-d_min)
    else:
        carte_prox = np.zeros((sh, sw), dtype=np.float32)

    # Fusion
    carte_poids = (carte_sal   * 1.0 +
                   carte_mov   * REACTIVITE +
                   carte_ctr   * 0.6 +
                   carte_inter * 0.4 +
                   carte_prox  * 0.5)
    mn, mx = carte_poids.min(), carte_poids.max()
    if mx > mn:
        carte_poids = (carte_poids - mn) / (mx - mn)

    # Z comme multiplicateur prioritaire
    if depth_map is not None:
        depth_small_n = cv2.resize(depth_map.astype(np.float32), (sw, sh),
                                   interpolation=cv2.INTER_AREA)
        valide = (depth_small_n > 100) & (depth_small_n < 15000)
        z_mult = np.full((sh, sw), 0.5, dtype=np.float32)
        if valide.any():
            d_min = depth_small_n[valide].min()
            d_max = depth_small_n[valide].max()
            if d_max > d_min:
                z_mult[valide] = 0.5 + 0.5*(1.0-(depth_small_n[valide]-d_min)/(d_max-d_min))
        carte_poids *= z_mult

    carte_lissee = cv2.GaussianBlur(carte_poids, (7, 7), 0)

    # Centre de l'image — candidat systématique
    pois = []
    cx_c, cy_c = w // 2, h // 2
    z_c        = z_ref_poi(cx_c, cy_c, depth_map)
    coh_c      = coherence_z_poi(cx_c, cy_c, depth_map)
    if coh_c > 0.4:
        pz_c   = poids_z(z_c, coh_c, depth_map)
        score_c = float(carte_lissee[min(int(cy_c/sy), sh-1),
                                     min(int(cx_c/sx), sw-1)])
        pois.append({
            "poi"        : (cx_c, cy_c),
            "poids"      : round(score_c * (0.5 + 0.5 * pz_c), 3),
            "saillance"  : 0.0, "mouvement": 0.0,
            "contraste"  : 0.0, "intersection": 0.0,
            "proximite"  : round(pz_c, 3),
            "poids_z"    : round(pz_c, 3),
            "coherence_z": round(coh_c, 3),
            "z_ref"      : z_c,
            "source"     : "centre",
            "bbox"       : efovea_bbox((cx_c, cy_c), h, w, depth_map),
        })

    # Maxima locaux
    rayon_sup  = max(sw, sh) // (n_max * 2)
    seuil      = INTENSITE_SEUIL * (1.0 - REACTIVITE * 0.3)
    carte_trav = carte_lissee.copy()

    for _ in range(n_max):
        max_val = float(carte_trav.max())
        if max_val < seuil:
            break
        _, _, _, max_loc = cv2.minMaxLoc(carte_trav)
        cx_s, cy_s = max_loc
        cx = int(cx_s * sx)
        cy = int(cy_s * sy)

        z_r   = z_ref_poi(cx, cy, depth_map)
        coh_z = coherence_z_poi(cx, cy, depth_map)
        pz    = poids_z(z_r, coh_z, depth_map)

        def _v(carte, cy_s=cy_s, cx_s=cx_s, sh=sh, sw=sw):
            return float(carte[min(cy_s, sh-1), min(cx_s, sw-1)])

        pois.append({
            "poi"        : (cx, cy),
            "poids"      : round(max_val * (0.5 + 0.5 * pz), 3),
            "saillance"  : round(_v(carte_sal),   3),
            "mouvement"  : round(_v(carte_mov),   3),
            "contraste"  : round(_v(carte_ctr),   3),
            "intersection": round(_v(carte_inter), 3),
            "proximite"  : round(_v(carte_prox),  3),
            "poids_z"    : round(pz, 3),
            "coherence_z": round(coh_z, 3),
            "z_ref"      : z_r,
            "source"     : "saillance",
            "bbox"       : efovea_bbox((cx, cy), h, w, depth_map),
        })
        cv2.circle(carte_trav, (cx_s, cy_s), rayon_sup, 0, -1)

    pois.sort(key=lambda p: p["poids"], reverse=True)
    return pois[:n_max]


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────
COULEUR_POI          = (255, 220, 0)
COULEUR_CHEMIN       = (255, 220, 0)
COULEUR_MIRE         = (0, 0, 220)
COULEUR_BBOX         = (0, 220, 255)
COULEUR_HEATMAP_ALPHA = 0.4
TAILLE_MIRE_MAX      = 28
TAILLE_MIRE_MIN      = 8


def dessiner_mire_poi(canvas, cx, cy, taille, couleur, epaisseur=1):
    cv2.line(canvas, (cx-taille, cy), (cx+taille, cy), couleur, epaisseur, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy-taille), (cx, cy+taille), couleur, epaisseur, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), max(3, taille//4), couleur, epaisseur, cv2.LINE_AA)


def dessiner_mire_centrale(canvas, cx, cy, taille=20):
    cv2.line(canvas, (cx-taille, cy), (cx+taille, cy), COULEUR_MIRE, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy-taille), (cx, cy+taille), COULEUR_MIRE, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 6, COULEUR_MIRE, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, COULEUR_MIRE, -1)


def dessiner_efovea(canvas, poi, bbox, saillance_val, z_ref=None):
    cx, cy       = poi
    x1, y1, x2, y2 = bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), COULEUR_BBOX, 2, cv2.LINE_AA)
    coin = 12
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(canvas, (px, py), (px+dx*coin, py), COULEUR_BBOX, 2)
        cv2.line(canvas, (px, py), (px, py+dy*coin), COULEUR_BBOX, 2)
    dessiner_mire_centrale(canvas, cx, cy)
    label = f"POI s:{saillance_val:.2f}"
    if z_ref and z_ref > 0:
        label += f" z:{z_ref:.0f}mm"
    cv2.putText(canvas, label, (x1, max(y1-6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COULEUR_BBOX, 1)


def dessiner_chemin_attentionnel(canvas, pois):
    if not pois:
        return
    poids_max = pois[0]["poids"]
    poids_min = pois[-1]["poids"] if len(pois) > 1 else poids_max

    def taille_mire(p):
        if poids_max == poids_min:
            return TAILLE_MIRE_MAX
        r = (p - poids_min) / (poids_max - poids_min)
        return int(TAILLE_MIRE_MIN + r * (TAILLE_MIRE_MAX - TAILLE_MIRE_MIN))

    # Tirets entre POIs
    for i in range(len(pois) - 1):
        p1 = pois[i]["poi"]; p2 = pois[i+1]["poi"]
        dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1:
            continue
        n_t = max(2, int(dist / 18))
        for j in range(n_t):
            t0 = j / n_t; t1 = t0 + 0.4 / n_t
            cv2.line(canvas,
                     (int(p1[0]+dx*t0), int(p1[1]+dy*t0)),
                     (int(p1[0]+dx*t1), int(p1[1]+dy*t1)),
                     COULEUR_CHEMIN, 2, cv2.LINE_AA)

    # Mires (ordre inverse — principal au-dessus)
    for i, pd in enumerate(reversed(pois)):
        cx, cy = pd["poi"]
        if i == len(pois) - 1:
            dessiner_efovea(canvas, (cx,cy), pd["bbox"],
                            pd["saillance"], pd["z_ref"])
        else:
            t = taille_mire(pd["poids"])
            dessiner_mire_poi(canvas, cx, cy, t, COULEUR_POI)
            cv2.putText(canvas, f"{pd['poids']:.2f}",
                        (cx+t+3, cy-t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COULEUR_POI, 1)


def dessiner_lignes_structures(canvas, lignes_v, lignes_h, intersections):
    for lv in lignes_v:
        x, y   = lv["position"]
        couleur = (0,255,255) if lv["type"]=="sol" else (255,255,0)
        cv2.line(canvas, (x, y), (x, max(0,y-80)), couleur, 1)
    for lh in lignes_h:
        x, y   = lh["position"]
        couleur = (0,255,255) if lh["type"]=="sol" else (255,255,0)
        cv2.line(canvas, (x,y), (x+lh["largeur"],y), couleur, 1)
    for inter in intersections:
        x, y = inter["position"]
        c    = (0,165,255) if inter["type"]=="coin_sol" else (255,165,0)
        t    = 8
        cv2.line(canvas, (x-t,y-t), (x+t,y+t), c, 2)
        cv2.line(canvas, (x+t,y-t), (x-t,y+t), c, 2)


def afficher_heatmap(frame, carte):
    carte_u8  = (carte * 255).astype(np.uint8)
    carte_big = cv2.resize(carte_u8, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame, 1-COULEUR_HEATMAP_ALPHA,
                           cv2.applyColorMap(carte_big, cv2.COLORMAP_JET),
                           COULEUR_HEATMAP_ALPHA, 0)



# ─────────────────────────────────────────────────────────────
#  SEGMENTATION ROI — Watershed guidé depth map
# ─────────────────────────────────────────────────────────────
def estimer_n_regions(roi_h: int, roi_w: int,
                      frame_h: int = H, frame_w: int = W) -> int:
    """
    Nombre de régions adaptatif selon la taille de la ROI.
    Petite ROI (<5% de la frame) → 1-2 régions
    Moyenne ROI (5-20%)          → 2-4 régions
    Grande ROI (>20%)            → 3-6 régions
    """
    frac = (roi_h * roi_w) / (frame_h * frame_w)
    if frac < 0.05:
        return 2
    elif frac < 0.20:
        return 4
    else:
        return 6


def segmenter_roi(roi_bgr: np.ndarray,
                  roi_depth: np.ndarray | None,
                  masque_z: np.ndarray,
                  poi_local: tuple) -> dict:
    """
    Segmentation watershed guidée par la depth map sur une ROI assainie.

    Marqueurs :
      - Fond certain  : pixels exclus par masque_z → marqueur 1
      - Noyau certain : zone autour du POI local avec Z très cohérent → marqueur 2+
      - Incertain     : tout le reste → laissé au watershed (marqueur 0)

    Retourne :
    {
        "labels"   : np.ndarray int32 — carte des labels par pixel
        "n_regions": int — nombre de régions trouvées
        "regions"  : list de dicts par région :
                     { "label", "aire", "centre", "z_median", "couleur_vis" }
        "mask_vis" : np.ndarray BGR — masque coloré pour visualisation
    }
    """
    h, w = roi_bgr.shape[:2]
    px, py = poi_local   # coordonnées du POI dans le repère ROI

    n_max = estimer_n_regions(h, w)

    # ── Préparation image pour watershed ─────────────────────────────────────
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # Gradient morphologique — met en évidence les bords entre régions
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(gray, kernel, iterations=1)
    dil     = cv2.dilate(gray, kernel, iterations=1)
    grad    = cv2.subtract(dil, erosion)

    # ── Construction des marqueurs ────────────────────────────────────────────
    markers = np.zeros((h, w), dtype=np.int32)

    # Marqueur 1 : fond certain = pixels exclus par l'assainissement Z
    fond_certain = (masque_z == 0)
    markers[fond_certain] = 1

    # Zone valide (non-fond)
    zone_valide = (masque_z > 0)

    if roi_depth is not None and zone_valide.any():
        depth_f = roi_depth.astype(np.float32)
        valides = depth_f[zone_valide & (depth_f > 100) & (depth_f < 15000)]

        if valides.size > 0:
            z_ref   = float(np.median(valides))
            sigma_z = float(np.std(valides))

            # Segmentation Z par k-means sur les profondeurs valides
            # On génère n_max marqueurs distincts selon les plans Z
            n_clusters = min(n_max, max(1, int(sigma_z / max(50, sigma_z/n_max))))
            n_clusters = max(1, min(n_clusters, n_max))

            depth_vals = depth_f[zone_valide & (depth_f > 100) & (depth_f < 15000)]
            if depth_vals.size > n_clusters and n_clusters > 1:
                # K-means 1D sur les profondeurs
                depth_col = depth_vals.reshape(-1, 1).astype(np.float32)
                _, labels_km, centres = cv2.kmeans(
                    depth_col, n_clusters,
                    None,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                    3, cv2.KMEANS_PP_CENTERS
                )
                # Ordre des clusters par profondeur croissante
                ordre = np.argsort(centres.flatten())
                remap = {old: new+2 for new, old in enumerate(ordre)}

                # Appliquer les labels kmeans comme marqueurs
                idx_valide = np.where(
                    zone_valide & (depth_f > 100) & (depth_f < 15000)
                )
                for i, (iy, ix) in enumerate(zip(idx_valide[0], idx_valide[1])):
                    markers[iy, ix] = remap[int(labels_km[i])]
            else:
                # Un seul cluster — marqueur 2 autour du POI
                rayon = min(h, w) // 6
                y1 = max(0, py - rayon); y2 = min(h, py + rayon)
                x1 = max(0, px - rayon); x2 = min(w, px + rayon)
                markers[y1:y2, x1:x2][zone_valide[y1:y2, x1:x2]] = 2
    else:
        # Pas de depth → marqueur unique autour du POI
        rayon = min(h, w) // 6
        y1 = max(0, py - rayon); y2 = min(h, py + rayon)
        x1 = max(0, px - rayon); x2 = min(w, px + rayon)
        markers[y1:y2, x1:x2] = 2

    # ── Watershed ─────────────────────────────────────────────────────────────
    roi_bgr_ws = roi_bgr.copy()
    roi_bgr_ws[masque_z == 0] = 0   # fond en noir

    cv2.watershed(roi_bgr_ws, markers)
    # markers == -1 : bords watershed

    # ── Extraction des régions ─────────────────────────────────────────────────
    labels_uniques = np.unique(markers)
    labels_uniques = labels_uniques[(labels_uniques > 1)]  # exclure fond(1) et bords(-1)

    # Palette de couleurs pour le mask coloré
    np.random.seed(42)
    palette = {
        lab: tuple(int(c) for c in np.random.randint(60, 240, 3))
        for lab in labels_uniques
    }

    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    regions  = []

    for lab in labels_uniques:
        region_mask = (markers == lab)
        aire        = int(region_mask.sum())
        if aire < 20:   # trop petit → ignoré
            continue

        # Centre de masse
        ys, xs = np.where(region_mask)
        cx_r   = int(xs.mean())
        cy_r   = int(ys.mean())

        # Profondeur médiane de la région
        z_med = None
        if roi_depth is not None:
            depth_f  = roi_depth.astype(np.float32)
            z_vals   = depth_f[region_mask & (depth_f > 100) & (depth_f < 15000)]
            z_med    = float(np.median(z_vals)) if z_vals.size > 0 else None

        couleur = palette[lab]
        mask_vis[region_mask] = couleur

        regions.append({
            "label"    : int(lab),
            "aire"     : aire,
            "centre"   : (cx_r, cy_r),
            "z_median" : round(z_med, 1) if z_med else None,
            "couleur_vis": couleur,
        })

    # Bords watershed en blanc
    mask_vis[markers == -1] = (255, 255, 255)

    # Tri par aire décroissante
    regions.sort(key=lambda r: r["aire"], reverse=True)

    return {
        "labels"   : markers,
        "n_regions": len(regions),
        "regions"  : regions,
        "mask_vis" : mask_vis,
    }



# ─────────────────────────────────────────────────────────────
#  SCORE SLAM — Utilisabilité d'une ROI pour la localisation
# ─────────────────────────────────────────────────────────────
# Seuils de qualification
SLAM_SEUIL_OK       = 0.55   # vert  — ROI fiable pour SLAM
SLAM_SEUIL_MAYBE    = 0.30   # orange — ROI borderline
SLAM_MIN_KEYPOINTS  = 8      # nombre minimum de keypoints détectables


def scorer_slam(roi_bgr: np.ndarray,
                masque_z: np.ndarray,
                sigma_z: float | None,
                coherence_z: float) -> dict:
    """
    Évalue l'utilisabilité d'une ROI pour le SLAM.

    Critères combinés :
      - texture       : richesse des gradients locaux (Laplacian variance)
      - coherence_z   : stabilité de la profondeur (passée depuis assainir_roi)
      - lisibilite    : plage photométrique exploitable (ni trop sombre ni saturée)
      - taille        : surface suffisante pour extraire des keypoints
      - n_keypoints   : nombre de coins Harris détectables

    Retourne :
    {
        "score"       : float [0,1]
        "tag"         : "OK" | "MAYBE" | "NON"
        "couleur_tag" : BGR tuple
        "detail"      : dict des composantes
        "n_keypoints" : int
    }
    """
    h, w = roi_bgr.shape[:2]

    # Zone valide uniquement
    valide = masque_z > 0
    if not valide.any():
        return {"score": 0.0, "tag": "NON",
                "couleur_tag": (0, 0, 180),
                "detail": {}, "n_keypoints": 0}

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Texture — variance du Laplacien sur la zone valide ────────────────
    lap     = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    lap_vals = lap[valide]
    texture  = float(np.var(lap_vals)) if lap_vals.size > 0 else 0.0
    # Normalisation empirique : variance > 500 → très texturé
    texture_n = float(min(1.0, texture / 500.0))

    # ── 2. Lisibilité photométrique ──────────────────────────────────────────
    gray_vals = gray[valide].astype(np.float32)
    lum_moy   = float(np.mean(gray_vals)) if gray_vals.size > 0 else 0.0
    lum_std   = float(np.std(gray_vals))  if gray_vals.size > 0 else 0.0
    # Optimal : luminance entre 40-220, std > 20 (pas d'aplat)
    lum_ok    = 1.0 - abs(lum_moy - 130) / 130.0   # pic à 130
    lum_ok    = max(0.0, lum_ok)
    std_ok    = min(1.0, lum_std / 60.0)
    lisibilite = 0.5 * lum_ok + 0.5 * std_ok

    # ── 3. Taille — fraction de pixels valides ───────────────────────────────
    frac_valide = float(valide.sum()) / (h * w)
    taille_n    = min(1.0, frac_valide / 0.5)   # 50% valide → score max

    # ── 4. Keypoints Harris ──────────────────────────────────────────────────
    corners   = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)
    corners   = np.clip(corners, 0, None)
    seuil_kp  = corners.max() * 0.01 if corners.max() > 0 else 1.0
    masque_kp = (corners > seuil_kp) & valide
    n_kp      = int(masque_kp.sum())
    kp_n      = min(1.0, n_kp / 30.0)   # 30 keypoints → score max

    # ── 5. Cohérence Z (déjà calculée) ──────────────────────────────────────
    coh_n = float(coherence_z)

    # ── Score final ──────────────────────────────────────────────────────────
    score = (texture_n  * 0.30 +
             lisibilite * 0.20 +
             taille_n   * 0.15 +
             kp_n       * 0.20 +
             coh_n      * 0.15)

    # Pénalité si trop peu de keypoints (SLAM impossible sans ancres)
    if n_kp < SLAM_MIN_KEYPOINTS:
        score *= 0.4

    score = float(min(1.0, score))

    if score >= SLAM_SEUIL_OK:
        tag, couleur = "OK",    (0, 200, 0)      # vert
    elif score >= SLAM_SEUIL_MAYBE:
        tag, couleur = "MAYBE", (0, 140, 255)    # orange
    else:
        tag, couleur = "NON",   (0, 0, 180)      # rouge sombre

    return {
        "score"      : round(score, 3),
        "tag"        : tag,
        "couleur_tag": couleur,
        "n_keypoints": n_kp,
        "detail"     : {
            "texture"   : round(texture_n,  3),
            "lisibilite": round(lisibilite, 3),
            "taille"    : round(taille_n,   3),
            "keypoints" : round(kp_n,       3),
            "coherence_z": round(coh_n,     3),
        }
    }


def dessiner_tag_slam(canvas: np.ndarray, bbox: tuple, slam: dict):
    """
    Dessine le badge SLAM sur la bbox d'un POI.
    SLAM✓ en vert, SLAM? en orange, rien si NON.
    """
    if slam["tag"] == "NON":
        return

    x1, y1, x2, y2 = bbox
    couleur = slam["couleur_tag"]
    symbole = "✓" if slam["tag"] == "OK" else "?"
    label   = f"SLAM{symbole} {slam['score']:.2f} ({slam['n_keypoints']}kp)"

    # Badge fond semi-transparent en bas de la bbox
    bx1, by1 = x1, y2 - 16
    bx2, by2 = x1 + len(label) * 7 + 6, y2
    overlay  = canvas.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), couleur, -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
    cv2.putText(canvas, label, (bx1 + 3, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

# ─────────────────────────────────────────────────────────────
#  PIPELINE OAK-D
# ─────────────────────────────────────────────────────────────
def creer_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb  = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(W, H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(FPS_ACQ)
    cam_rgb.initialControl.setManualFocus(120)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    for mono, socket in [(mono_l, dai.CameraBoardSocket.CAM_B),
                         (mono_r, dai.CameraBoardSocket.CAM_C)]:
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono.setBoardSocket(socket)
        mono.setFps(FPS_ACQ)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(W, H)
    stereo.setSubpixel(True)

    cfg = stereo.initialConfig.get()
    cfg.postProcessing.speckleFilter.enable            = True
    cfg.postProcessing.speckleFilter.speckleRange      = 60
    cfg.postProcessing.temporalFilter.enable           = True
    cfg.postProcessing.temporalFilter.alpha            = 0.4
    cfg.postProcessing.spatialFilter.enable            = True
    cfg.postProcessing.spatialFilter.holeFillingRadius = 2
    cfg.postProcessing.spatialFilter.numIterations     = 1
    cfg.postProcessing.thresholdFilter.minRange        = 200
    cfg.postProcessing.thresholdFilter.maxRange        = 8000
    stereo.initialConfig.set(cfg)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    ctrl_in = pipeline.create(dai.node.XLinkIn)
    ctrl_in.setStreamName("control")
    ctrl_in.out.link(cam_rgb.inputControl)

    for name, src in [("rgb", cam_rgb.preview), ("depth", stereo.depth)]:
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        src.link(xout.input)

    return pipeline


# ─────────────────────────────────────────────────────────────
#  THREADS ACQUISITION
# ─────────────────────────────────────────────────────────────
def thread_acquisition_oak(device, frame_queue, stop_event):
    q_rgb   = device.getOutputQueue("rgb",   maxSize=1, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
    q_ctrl  = device.getInputQueue("control")

    dernier_af     = datetime.now()
    lenspos_actuel = lenspos_cible = 120
    EMA_ALPHA      = 0.25 if IS_PI else 0.35
    depth_ema      = None

    while not stop_event.is_set():
        in_rgb   = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        if in_rgb is None or in_depth is None:
            continue

        frame_bgr = in_rgb.getCvFrame()
        depth_raw = in_depth.getFrame()
        ts        = datetime.now()

        lp = in_rgb.getLensPosition()
        if lp != -1 and abs(lp - lenspos_cible) > 2:
            continue

        mask = depth_raw > 0
        if depth_ema is None:
            depth_ema = depth_raw.astype(np.float32)
        else:
            depth_ema[mask] = (EMA_ALPHA * depth_raw[mask].astype(np.float32)
                               + (1 - EMA_ALPHA) * depth_ema[mask])
        depth_final = np.clip(depth_ema, 0, 65535).astype(np.uint16)

        if (ts - dernier_af).total_seconds() >= AF_PERIODE_S:
            dist_mm = distance_mediane_centrale(depth_final)
            nlp     = distance_vers_lenspos(dist_mm)
            if abs(nlp - lenspos_actuel) > 5:
                ctrl = dai.CameraControl()
                ctrl.setManualFocus(nlp)
                q_ctrl.send(ctrl)
                lenspos_actuel = lenspos_cible = nlp
                depth_ema      = None
            dernier_af = ts

        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put((frame_bgr, depth_final, ts))


def thread_acquisition_webcam(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_ACQ)
    if not cap.isOpened():
        print("[ERREUR] Webcam inaccessible")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: continue
        fh, fw = frame.shape[:2]
        frame  = frame[(fh-400)//2:(fh-400)//2+400, :] if fh >= 400 \
                 else cv2.resize(frame, (W, H))
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put((frame, None, datetime.now()))
    cap.release()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=QUEUE_MAX)
    stop_event  = threading.Event()

    if USE_OAKD:
        pipeline = creer_pipeline()
        device   = dai.Device(pipeline)
        t_acq    = threading.Thread(target=thread_acquisition_oak,
                                    args=(device, frame_queue, stop_event), daemon=True)
    else:
        t_acq = threading.Thread(target=thread_acquisition_webcam,
                                 args=(frame_queue, stop_event), daemon=True)
    t_acq.start()

    mode_analyse = False
    derniere_frame = avant_derniere = derniere_depth = resultats_analyse = None
    show_heatmap = False

    print("\n=== Retina Lab — Shirka_001 ===")
    print("ESPACE : analyser  H : heatmap  Q : quitter\n")

    while not stop_event.is_set():
        try:
            frame, depth, _ = frame_queue.get(timeout=0.05)
            avant_derniere  = derniere_frame
            derniere_frame  = frame
            derniere_depth  = depth
        except queue.Empty:
            pass

        if mode_analyse and resultats_analyse is not None:
            affichage = resultats_analyse
        elif derniere_frame is not None:
            affichage = derniere_frame.copy()
            cv2.putText(affichage, "ESPACE:analyser  H:heatmap  Q:quitter",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        else:
            continue

        cv2.imshow("Retina Lab — Shirka", affichage)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            stop_event.set()

        elif key == ord('h'):
            show_heatmap = not show_heatmap

        elif key == ord(' '):
            if not mode_analyse and derniere_frame is not None:
                print("Analyse rétine en cours...")

                pois = detecter_pois(derniere_frame, avant_derniere,
                                     derniere_depth, n_max=6)

                plans = detecter_plans_principaux(derniere_depth)
                sol, plafond = plans["sol"], plans["plafond"]

                lignes_v = detecter_lignes_verticales(
                    derniere_depth, sol, plafond) if derniere_depth is not None else []
                lignes_h = detecter_lignes_horizontales(
                    derniere_depth, sol, plafond) if derniere_depth is not None else []
                intersections = detecter_intersections_3d(
                    derniere_depth, sol, plafond,
                    lignes_v, lignes_h) if derniere_depth is not None else []

                affichage = derniere_frame.copy()
                if show_heatmap:
                    fs = cv2.resize(derniere_frame,
                                    (max(1,W//DIV_RETINE), max(1,H//DIV_RETINE)),
                                    interpolation=cv2.INTER_AREA)
                    affichage = afficher_heatmap(affichage, carte_saillance_gem(fs))

                if pois:
                    for p in pois:
                        x1, y1, x2, y2 = p["bbox"]
                        roi_bgr   = derniere_frame[y1:y2, x1:x2]
                        roi_depth = derniere_depth[y1:y2, x1:x2] \
                                    if derniere_depth is not None else None
                        roi_s, masque, z_r, sig_z = assainir_roi(roi_bgr, roi_depth)
                        p["roi_assainie"] = roi_s
                        p["masque_z"]     = masque
                        p["sigma_z"]      = round(sig_z, 1) if sig_z else None

                    dessiner_chemin_attentionnel(affichage, pois)
                    dessiner_lignes_structures(affichage, lignes_v, lignes_h, intersections)
                    # Tags SLAM sur les POIs scorés
                    if pois[0].get("slam"):
                        dessiner_tag_slam(affichage, pois[0]["bbox"], pois[0]["slam"])

                    print(f"{len(pois)} POI(s) :")
                    for i, p in enumerate(pois):
                        z_str = f"z:{p['z_ref']:.0f}mm" if p["z_ref"] else "z:N/A"
                        sig_str = f"σz:{p['sigma_z']}mm" if p.get("sigma_z") else ""
                        print(f"  #{i+1} [{p['source']}] {p['poi']}  "
                              f"poids:{p['poids']:.2f}  sal:{p['saillance']:.2f}  "
                              f"mov:{p['mouvement']:.2f}  ctr:{p['contraste']:.2f}  "
                              f"pz:{p['poids_z']:.2f}  coh:{p['coherence_z']:.2f}  "
                              f"{z_str}  {sig_str}")
                    print(f"  Structures: {len(lignes_v)}V  "
                          f"{len(lignes_h)}H  {len(intersections)} intersections")
                else:
                    cv2.putText(affichage, "Aucun POI",
                                (W//2-40, H//2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,0,200), 2)
                    print("Aucun POI détecté")

                # Segmentation + debug ROI POI #1
                if pois and pois[0].get("roi_assainie") is not None:
                    p0      = pois[0]
                    x1, y1, x2, y2 = p0["bbox"]
                    # POI en coordonnées locales ROI
                    px_loc  = min(max(0, p0["poi"][0] - x1), (x2-x1)-1)
                    py_loc  = min(max(0, p0["poi"][1] - y1), (y2-y1)-1)

                    roi_depth_clip = derniere_depth[y1:y2, x1:x2]                                      if derniere_depth is not None else None

                    seg = segmenter_roi(
                        p0["roi_assainie"],
                        roi_depth_clip,
                        p0["masque_z"],
                        (px_loc, py_loc)
                    )
                    p0["segmentation"] = seg

                    # Score SLAM
                    slam = scorer_slam(
                        p0["roi_assainie"],
                        p0["masque_z"],
                        p0.get("sigma_z"),
                        p0.get("coherence_z", 0.5)
                    )
                    p0["slam"] = slam

                    # Mask coloré — affiché en debug
                    mask_vis = cv2.resize(seg["mask_vis"], (320, 200),
                                          interpolation=cv2.INTER_NEAREST)
                    z_r   = p0["z_ref"]
                    sig_z = p0["sigma_z"]
                    info  = (f"Seg POI#1  {seg['n_regions']} région(s)"
                             f"  z:{z_r:.0f}mm  σ:{sig_z}mm"
                             if z_r else
                             f"Seg POI#1  {seg['n_regions']} région(s) (webcam)")
                    cv2.putText(mask_vis, info, (4, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)

                    # Annoter les centres des régions
                    sx = 320 / max(1, x2-x1)
                    sy = 200 / max(1, y2-y1)
                    for reg in seg["regions"]:
                        cx_r = int(reg["centre"][0] * sx)
                        cy_r = int(reg["centre"][1] * sy)
                        cv2.circle(mask_vis, (cx_r, cy_r), 4, (255,255,255), -1)
                        z_label = f"{reg['z_median']:.0f}mm" if reg["z_median"] else ""
                        cv2.putText(mask_vis, z_label,
                                    (cx_r+5, cy_r),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

                    cv2.imshow("Segmentation — POI #1", mask_vis)
                    print(f"  Segmentation POI#1 : {seg['n_regions']} région(s)")
                    for reg in seg["regions"]:
                        print(f"    label:{reg['label']}  "
                              f"aire:{reg['aire']}px²  "
                              f"centre:{reg['centre']}  "
                              f"z:{reg['z_median']}mm")
                    if p0.get("slam"):
                        sl = p0["slam"]
                        print(f"  SLAM POI#1 : tag:{sl['tag']}  "
                              f"score:{sl['score']}  kp:{sl['n_keypoints']}  "
                              f"tex:{sl['detail']['texture']}  "
                              f"lum:{sl['detail']['lisibilite']}  "
                              f"coh_z:{sl['detail']['coherence_z']}")

                # Debug saillance | mouvement
                fs = cv2.resize(derniere_frame,
                                (max(1,W//DIV_RETINE), max(1,H//DIV_RETINE)),
                                interpolation=cv2.INTER_AREA)
                sal_u8 = cv2.resize((carte_saillance_gem(fs)*255).astype(np.uint8),
                                    (W//2, H))
                mov_u8 = cv2.resize(
                    (carte_mouvement(derniere_frame, avant_derniere)*255).astype(np.uint8),
                    (W//2, H))
                debug_img = np.hstack([
                    cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET),
                    cv2.applyColorMap(mov_u8, cv2.COLORMAP_HOT)
                ])
                cv2.putText(debug_img, "Saillance GEM", (6,15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(debug_img, "Mouvement", (W//2+6,15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.imshow("Debug — Saillance | Mouvement", debug_img)

                resultats_analyse = affichage
                mode_analyse      = True

            else:
                mode_analyse = False
                resultats_analyse = None
                for win in ["Debug — Saillance | Mouvement",
                            "Segmentation — POI #1"]:
                    try: cv2.destroyWindow(win)
                    except Exception: pass
                print("Retour temps réel")

    if USE_OAKD:
        device.close()
    cv2.destroyAllWindows()
    print("Arrêt.")
